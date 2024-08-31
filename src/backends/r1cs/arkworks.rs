use crate::backends::r1cs::{LinearCombination as NoNameLinearCombination, R1csBn254Field};
use crate::backends::{
    r1cs::{GeneratedWitness, R1CS},
    BackendField,
};
use crate::inputs::{parse_inputs, JsonInputs};
use crate::witness::CompiledCircuit;
use ark_bls12_381::Fr;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, LinearCombination,
    SynthesisError, Variable,
};
use num_bigint::BigUint;

use crate::circuit_writer::CircuitWriter;
use crate::compiler::{typecheck_next_file, Sources};
use crate::type_checker::TypeChecker;

pub struct NoNameCircuit<BF: BackendField> {
    compiled_circuit: CompiledCircuit<R1CS<BF>>,
    witness: GeneratedWitness<BF>,
}

impl<BF: BackendField> ConstraintSynthesizer<BF> for NoNameCircuit<BF> {
    fn generate_constraints(self, cs: ConstraintSystemRef<BF>) -> Result<(), SynthesisError> {
        let public_io_length = self.compiled_circuit.circuit.backend.public_inputs.len()
            + self.compiled_circuit.circuit.backend.public_outputs.len();

        // arkworks assigns by default the 1 constant
        // assumes witness is: [1, public_outputs, public_inputs, private_inputs, aux]
        let witness_size = self.witness.witness.len();
        for idx in 1..witness_size {
            let value: BigUint = Into::into(self.witness.witness[idx]);
            let field_element = BF::from(value);
            if idx <= public_io_length {
                cs.new_input_variable(|| Ok(field_element))?;
            } else {
                cs.new_witness_variable(|| Ok(field_element))?;
            }
        }

        let make_index = |index| {
            if index <= public_io_length {
                match index == 0 {
                    true => Variable::One,
                    false => Variable::Instance(index),
                }
            } else {
                Variable::Witness(index - (public_io_length + 1))
            }
        };

        let make_lc = |lc_data: NoNameLinearCombination<BF>| {
            let mut lc = LinearCombination::<BF>::zero();
            for (cellvar, coeff) in lc_data.terms.into_iter() {
                let idx = make_index(cellvar.index);
                let coeff = BF::from(Into::<BigUint>::into(coeff));
                lc += (coeff, idx)
            }

            // add constant
            let constant = BF::from(Into::<BigUint>::into(lc_data.constant));
            lc += (constant, make_index(0));
            lc
        };

        for constraint in self.compiled_circuit.circuit.backend.constraints {
            cs.enforce_constraint(
                make_lc(constraint.a),
                make_lc(constraint.b),
                make_lc(constraint.c),
            )?;
        }

        Ok(())
    }
}

pub const SIMPLE_ADDITION: &str = "fn main(pub public_input: Field, private_input: Field) {
    let xx = private_input + public_input;
    let yy = private_input * public_input;
    assert_eq(xx, yy);
}
";

pub const WITH_PUBLIC_OUTPUT_ARRAY: &str =
    "fn main(pub public_input: [Field; 2], private_input: [Field; 2]) -> [Field; 2]{
    let xx = private_input[0] + public_input[0];
    let yy = private_input[1] * public_input[1];
    assert_eq(yy, xx);
    return [xx, yy];
}";

pub const CONSTRAINT_FAILURE: &str = "
    fn main(pub public_input: Field, private_input: Field) {
        let xx = private_input + public_input;
        let yy = private_input * public_input;
        assert_eq(xx, yy + 1); // This should fail
    }
";

pub const FIELD_ADDITION_SUBTRACTION: &str = "
    fn main(pub public_input: Field, private_input: Field) {
        let xx = public_input + private_input;
        let yy = xx - private_input;
        assert_eq(public_input, yy);
    }
";

pub const FIELD_MULTIPLICATION: &str = "
    fn main(pub public_input: Field, private_input: Field) {
        let xx = public_input * private_input;
        let yy = xx;
        assert_eq(yy, public_input * private_input);
    }
";

pub const BOOLEAN_AND_OR: &str = "fn main(pub public_input: Field, private_input: Field) {
    let xx = public_input * private_input;
    let yy = (public_input + private_input) - (public_input * private_input);
    assert_eq(xx + yy, public_input + private_input);
}
";

pub const BOOLEAN_NOT: &str = "
    fn main(pub public_input: Field, private_input: Field) {
        let xx = private_input * (1 - public_input);
        assert_eq(xx + public_input, 1);
    }
";
pub const ASM_SNAPSHOT_TEST: &str = r#"fn main(pub public_input: Field, private_input: Field) -> Field {
    let xx = private_input + public_input;
    assert_eq(xx, 2);
    let yy = xx + 6;
    return yy;
}"#;

pub const ASM_SNAPSHOT_TEST_ADDITION: &str = r#"fn main(pub public_input: Field, private_input: Field) -> Field {
    let xx = private_input + public_input;
    return xx;
}"#;

pub const ASM_SNAPSHOT_TEST_MULTIPLICATION: &str = r#"fn main(pub public_input: Field, private_input: Field) -> Field {
    let xx = private_input * public_input;
    return xx;
}"#;

pub const ASM_SNAPSHOT_TEST_BOOL: &str = r#"fn main(pub public_input: Field, private_input: Field) -> Field {
    let diff = public_input - private_input;
    let is_greater = diff * diff; // Ensures a non-zero value if true, zero if false
    return is_greater;
}"#;

pub const TEST_BUILTIN_ASSERT_TRUE: &str = r#"fn main(input: Field) { assert(input == input); }"#;

pub const TEST_BUILTIN_ASSERT_FALSE: &str = r#"fn main(input: Field) { assert(input == 0); }"#;

pub const TEST_BUILTIN_ASSERT_EQ_TRUE: &str = r#"fn main(input: Field) {
    let doubled = input + input;
    let squared = input * input;
    assert_eq(doubled, input + input);
    assert_eq(squared, input * input);
}"#;

pub const TEST_BUILTIN_ASSERT_EQ_FALSE: &str = r#"fn main(input: Field) { assert_eq(1, 0); }"#;

pub fn compile_source_code<BF: BackendField>(
    code: &str,
) -> Result<CompiledCircuit<R1CS<BF>>, crate::error::Error> {
    let mut sources = Sources::new();

    // parse the transitive dependency
    let mut tast = TypeChecker::<R1CS<BF>>::new();
    let mut node_id = 0;
    node_id = typecheck_next_file(
        &mut tast,
        None,
        &mut sources,
        "main.no".to_string(),
        code.to_string(),
        node_id,
    )
    .unwrap();
    let r1cs = R1CS::<BF>::new();
    // compile
    CircuitWriter::generate_circuit(tast, r1cs)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        backends::{r1cs::R1csBn254Field, Backend},
        error::ErrorKind,
        inputs::{parse_inputs, JsonInputs},
    };
    use ark_bn254::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_arkworks_cs_is_satisfied() {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(SIMPLE_ADDITION).unwrap();
        let inputs_public = r#"{"public_input": "2"}"#;
        let inputs_private = r#"{"private_input": "2"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();
        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();

        let noname_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };

        let cs = ConstraintSystem::<Fr>::new_ref();
        noname_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_arkworks_cs_is_satisfied_array() {
        let compiled_circuit =
            compile_source_code::<R1csBn254Field>(WITH_PUBLIC_OUTPUT_ARRAY).unwrap();
        let inputs_public = r#"{"public_input": ["2", "5"]}"#;
        let inputs_private = r#"{"private_input": ["8", "2"]}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();
        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();
        let noname_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };

        let cs = ConstraintSystem::<Fr>::new_ref();
        noname_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_constraint_failure() {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(CONSTRAINT_FAILURE).unwrap();
        let inputs_public = r#"{"public_input": "2"}"#;
        let inputs_private = r#"{"private_input": "2"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();

        if let Err(err) = compiled_circuit.generate_witness(json_public, json_private) {
            println!("Failed to generate witness as expected: {:?}", err);
        } else {
            panic!("Expected witness generation to fail, but it succeeded");
        }
    }

    #[test]
    fn test_field_addition_subtraction() {
        let compiled_circuit =
            compile_source_code::<R1csBn254Field>(FIELD_ADDITION_SUBTRACTION).unwrap();
        let inputs_public = r#"{"public_input": "5"}"#;
        let inputs_private = r#"{"private_input": "3"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();
        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();

        let no_name_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        no_name_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_field_multiplication() {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(FIELD_MULTIPLICATION).unwrap();
        let inputs_public = r#"{"public_input": "6"}"#;
        let inputs_private = r#"{"private_input": "7"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();

        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();

        let no_name_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        no_name_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_boolean_and_or() {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(BOOLEAN_AND_OR).unwrap();
        let inputs_public = r#"{"public_input": "1"}"#;
        let inputs_private = r#"{"private_input": "0"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();

        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();

        let no_name_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        no_name_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_boolean_not() {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(BOOLEAN_NOT).unwrap();
        let inputs_public = r#"{"public_input": "0"}"#;
        let inputs_private = r#"{"private_input": "1"}"#;

        let json_public = parse_inputs(inputs_public).unwrap();
        let json_private = parse_inputs(inputs_private).unwrap();

        let generated_witness = compiled_circuit
            .generate_witness(json_public, json_private)
            .unwrap();

        let no_name_circuit = NoNameCircuit {
            compiled_circuit,
            witness: generated_witness,
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        no_name_circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }
    #[test]
    fn test_asm_snapshot() -> miette::Result<()> {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(ASM_SNAPSHOT_TEST)?;

        let asm_output = compiled_circuit
            .circuit
            .backend
            .generate_asm(&Sources::new(), false);

        let expected_asm_output = r#"@ noname.0.7.0

2 == (v_2 + v_3) * (1)
v_2 + v_3 + 6 == (v_1) * (1)"#;

        // Compare the generated ASM output with the expected one
        assert_eq!(asm_output.trim(), expected_asm_output.trim());

        Ok(())
    }
    #[test]
    fn test_asm_snapshot_addition() -> miette::Result<()> {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(ASM_SNAPSHOT_TEST_ADDITION)?;

        let asm_output = compiled_circuit
            .circuit
            .backend
            .generate_asm(&Sources::new(), false);

        let expected_asm_output = r#"@ noname.0.7.0

v_2 + v_3 == (v_1) * (1)"#;

        assert_eq!(asm_output.trim(), expected_asm_output.trim());

        Ok(())
    }

    #[test]
    fn test_asm_snapshot_multiplication() -> miette::Result<()> {
        let compiled_circuit =
            compile_source_code::<R1csBn254Field>(ASM_SNAPSHOT_TEST_MULTIPLICATION)?;

        let asm_output = compiled_circuit
            .circuit
            .backend
            .generate_asm(&Sources::new(), false);

        let expected_asm_output = r#"@ noname.0.7.0

v_4 == (v_3) * (v_2)
v_4 == (v_1) * (1)"#;

        assert_eq!(asm_output.trim(), expected_asm_output.trim());

        Ok(())
    }

    #[test]
    fn test_asm_snapshot_bool() -> miette::Result<()> {
        let compiled_circuit = compile_source_code::<R1csBn254Field>(ASM_SNAPSHOT_TEST_BOOL)?;

        let asm_output = compiled_circuit
            .circuit
            .backend
            .generate_asm(&Sources::new(), false);

        let expected_asm_output = r#"@ noname.0.7.0

v_4 == (v_2 + -1 * v_3) * (v_2 + -1 * v_3)
v_4 == (v_1) * (1)"#;

        assert_eq!(asm_output.trim(), expected_asm_output.trim());

        Ok(())
    }
}

#[test]
fn test_builtin_assert_true() {
    let compiled_circuit = compile_source_code::<R1csBn254Field>(TEST_BUILTIN_ASSERT_TRUE).unwrap();

    // Providing necessary inputs
    let inputs_private = r#"{"input": "1"}"#; // This should pass the assert

    let json_private = parse_inputs(inputs_private).unwrap();

    let generated_witness = compiled_circuit
        .generate_witness(JsonInputs::default(), json_private)
        .unwrap();

    let no_name_circuit = NoNameCircuit {
        compiled_circuit,
        witness: generated_witness,
    };
    let cs = ConstraintSystem::<ark_bn254::Fr>::new_ref();
    no_name_circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());
}

#[test]
#[should_panic(expected = "InvalidWitness")]
fn test_builtin_assert_false() {
    let compiled_circuit =
        compile_source_code::<R1csBn254Field>(TEST_BUILTIN_ASSERT_FALSE).unwrap();

    let inputs_private = r#"{"input": "1"}"#;

    let json_private = parse_inputs(inputs_private).unwrap();

    let generated_witness = compiled_circuit
        .generate_witness(JsonInputs::default(), json_private)
        .unwrap();

    let no_name_circuit = NoNameCircuit {
        compiled_circuit,
        witness: generated_witness,
    };
    let cs = ConstraintSystem::<ark_bn254::Fr>::new_ref();
    no_name_circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());
}

#[test]
fn test_builtin_assert_eq_true() {
    let compiled_circuit =
        compile_source_code::<R1csBn254Field>(TEST_BUILTIN_ASSERT_EQ_TRUE).unwrap();

    // Providing necessary inputs
    let inputs_private = r#"{"input": "1"}"#; // This should pass the assert_eq

    let json_private = parse_inputs(inputs_private).unwrap();

    let generated_witness = compiled_circuit
        .generate_witness(JsonInputs::default(), json_private)
        .unwrap();

    let no_name_circuit = NoNameCircuit {
        compiled_circuit,
        witness: generated_witness,
    };
    let cs = ConstraintSystem::<ark_bn254::Fr>::new_ref();
    no_name_circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());
}

#[test]
#[should_panic(expected = "AssertionFailed")]
fn test_builtin_assert_eq_false() {
    let compiled_circuit =
        compile_source_code::<R1csBn254Field>(TEST_BUILTIN_ASSERT_EQ_FALSE).unwrap();
    let generated_witness = compiled_circuit
        .generate_witness(JsonInputs::default(), JsonInputs::default())
        .unwrap();
    let no_name_circuit = NoNameCircuit {
        compiled_circuit,
        witness: generated_witness,
    };
    let cs = ConstraintSystem::<ark_bn254::Fr>::new_ref();
    no_name_circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());
}

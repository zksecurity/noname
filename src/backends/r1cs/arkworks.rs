use crate::backends::r1cs::LinearCombination as NoNameLinearCombination;
use crate::backends::{
    r1cs::{GeneratedWitness, R1CS},
    BackendField,
};
use crate::mast::Mast;
use crate::witness::CompiledCircuit;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, LinearCombination, SynthesisError, Variable,
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

    let mut mast = Mast::new(tast);
    mast.monomorphize()?;
    let r1cs = R1CS::<BF>::new();
    // compile
    CircuitWriter::generate_circuit(mast, r1cs)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{backends::r1cs::R1csBn254Field, inputs::parse_inputs};
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
}

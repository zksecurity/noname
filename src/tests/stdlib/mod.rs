mod comparator;
mod mimc;
mod multiplexer;
mod uints;

use std::{path::Path, str::FromStr};

use crate::{
    backends::r1cs::{R1csBn254Field, R1CS},
    circuit_writer::CircuitWriter,
    compiler::{typecheck_next_file, Sources},
    error::Result,
    inputs::parse_inputs,
    mast,
    stdlib::{init_stdlib_dep, STDLIB_DIRECTORY},
    type_checker::TypeChecker,
    witness::CompiledCircuit,
};

fn test_stdlib(
    path: &str,
    asm_path: Option<&str>,
    public_inputs: &str,
    private_inputs: &str,
    expected_public_output: Vec<&str>,
) -> Result<CompiledCircuit<R1CS<R1csBn254Field>>> {
    let root = env!("CARGO_MANIFEST_DIR");
    let prefix_path = Path::new(root).join("src/tests/stdlib");

    // read noname file
    let code = std::fs::read_to_string(prefix_path.clone().join(path)).unwrap();

    let compiled_circuit = test_stdlib_code(
        &code,
        asm_path,
        public_inputs,
        private_inputs,
        expected_public_output,
    )?;

    Ok(compiled_circuit)
}

fn test_stdlib_code(
    code: &str,
    asm_path: Option<&str>,
    public_inputs: &str,
    private_inputs: &str,
    expected_public_output: Vec<&str>,
) -> Result<CompiledCircuit<R1CS<R1csBn254Field>>> {
    let r1cs = R1CS::new();
    let root = env!("CARGO_MANIFEST_DIR");

    // parse inputs
    let public_inputs = parse_inputs(public_inputs).unwrap();
    let private_inputs = parse_inputs(private_inputs).unwrap();

    // compile
    let mut sources = Sources::new();
    let mut tast = TypeChecker::new();
    let mut node_id = 0;
    node_id = init_stdlib_dep(&mut sources, &mut tast, node_id, STDLIB_DIRECTORY);

    let this_module = None;
    let _node_id = typecheck_next_file(
        &mut tast,
        this_module,
        &mut sources,
        "test.no".to_string(),
        code.to_string(),
        node_id,
    )
    .unwrap();

    let mast = mast::monomorphize(tast)?;
    let compiled_circuit = CircuitWriter::generate_circuit(mast, r1cs)?;

    // this should check the constraints
    let generated_witness =
        compiled_circuit.generate_witness(public_inputs.clone(), private_inputs.clone())?;

    let expected_public_output = expected_public_output
        .iter()
        .map(|x| crate::backends::r1cs::R1csBn254Field::from_str(x).unwrap())
        .collect::<Vec<_>>();

    if generated_witness.outputs != expected_public_output {
        eprintln!("obtained by executing the circuit:");
        generated_witness
            .outputs
            .iter()
            .for_each(|x| eprintln!("- {x}"));
        eprintln!("passed as output by the verifier:");
        expected_public_output
            .iter()
            .for_each(|x| eprintln!("- {x}"));
        panic!("Obtained output does not match expected output");
    }

    // check the ASM
    if asm_path.is_some() && compiled_circuit.circuit.backend.num_constraints() < 100 {
        let prefix_asm = Path::new(root).join("src/tests/stdlib/");
        let expected_asm =
            std::fs::read_to_string(prefix_asm.clone().join(asm_path.unwrap())).unwrap();
        let obtained_asm = compiled_circuit.asm(&Sources::new(), false);

        if obtained_asm != expected_asm {
            eprintln!("obtained:");
            eprintln!("{obtained_asm}");
            eprintln!("expected:");
            eprintln!("{expected_asm}");
            panic!("Obtained ASM does not match expected ASM");
        }
    }

    Ok(compiled_circuit)
}

use std::{collections::HashMap, path::Path};

use crate::{ast::CellValues, field::Field, prover::compile};

fn test_file(file_name: &str, args: HashMap<&str, CellValues>, expected_public_output: Vec<Field>) {
    let version = env!("CARGO_MANIFEST_DIR");
    let prefix = Path::new(version).join("data");

    // read noname file
    let code = std::fs::read_to_string(prefix.clone().join(format!("{file_name}.no"))).unwrap();
    let asm = std::fs::read_to_string(prefix.clone().join(format!("{file_name}.asm"))).unwrap();

    // compile
    let (circuit, prover_index, verifier_index) = compile(&code, false).unwrap();

    // check compiled ASM
    assert_eq!(circuit, asm);

    // create proof
    let (proof, full_public_inputs, public_output) = prover_index.prove(args, false).unwrap();
    assert_eq!(public_output, expected_public_output);

    // verify proof
    verifier_index.verify(full_public_inputs, proof).unwrap();
}

#[test]
fn test_arithmetic() {
    let mut args = HashMap::new();
    args.insert("public_input", CellValues::new(vec![1u32.into()]));
    args.insert("private_input", CellValues::new(vec![1u32.into()]));

    test_file("arithmetic", args, vec![]);
}

#[test]
fn test_public_output() {
    let mut args = HashMap::new();
    args.insert("public_input", CellValues::new(vec![1.into()]));
    args.insert("private_input", CellValues::new(vec![1.into()]));

    test_file("public_output", args, vec![8u32.into()]);
}

#[test]
fn test_poseidon() {
    let private_input = [1.into(), 1.into()];
    let digest = crate::helpers::poseidon(private_input.clone());

    let mut args = HashMap::new();
    args.insert("public_input", CellValues::new(vec![digest]));
    args.insert("private_input", CellValues::new(private_input.to_vec()));

    test_file("poseidon", args, vec![]);
}

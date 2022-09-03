use std::path::Path;

use crate::{
    field::Field,
    inputs::{parse_inputs, ExtField},
    prover::compile_and_prove,
};

fn test_file(
    file_name: &str,
    public_inputs: &str,
    private_inputs: &str,
    expected_public_output: Vec<Field>,
) {
    let version = env!("CARGO_MANIFEST_DIR");
    let prefix = Path::new(version).join("data");

    // read noname file
    let code = std::fs::read_to_string(prefix.clone().join(format!("{file_name}.no"))).unwrap();
    let asm = std::fs::read_to_string(prefix.clone().join(format!("{file_name}.asm"))).unwrap();

    // compile
    let (prover_index, verifier_index) = compile_and_prove(&code).unwrap();

    // check compiled ASM
    assert_eq!(prover_index.asm(false), asm);

    // parse inputs
    let public_inputs = parse_inputs(public_inputs).unwrap();
    let private_inputs = parse_inputs(private_inputs).unwrap();

    // create proof
    let (proof, full_public_inputs, public_output) = prover_index
        .prove(public_inputs, private_inputs, false)
        .unwrap();
    assert_eq!(public_output, expected_public_output);

    // verify proof
    verifier_index.verify(full_public_inputs, proof).unwrap();
}

#[test]
fn test_arithmetic() {
    let public_inputs = r#"{"public_input": ["1"]}"#;
    let private_inputs = r#"{"private_input": ["1"]}"#;

    println!("public inputs: {:?}", public_inputs);
    println!("private inputs: {:?}", private_inputs);

    test_file("arithmetic", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_public_output() {
    let public_inputs = r#"{"public_input": ["1"]}"#;
    let private_inputs = r#"{"private_input": ["1"]}"#;

    test_file(
        "public_output",
        public_inputs,
        private_inputs,
        vec![8u32.into()],
    );
}

#[test]
fn test_poseidon() {
    let private_inputs = r#"{"private_input": ["1", "1"]}"#;
    let private_input = [1.into(), 1.into()];
    let digest = crate::helpers::poseidon(private_input.clone());
    let digest_dec = digest.to_dec_string();
    assert_eq!(
        "3654913405619483358804575553468071097765421484960111776885779739261304758583",
        digest_dec
    );

    let public_inputs = &format!(r#"{{"public_input": ["{digest_dec}"]}}"#);

    test_file("poseidon", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_bool() {
    let private_inputs = r#"{"private_input": ["0"]}"#;
    let public_inputs = r#"{"public_input": ["1"]}"#;

    test_file("bool", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_mutable() {
    let private_inputs = r#"{"x": ["2"], "y": ["3"]}"#;
    let public_inputs = r#"{}"#;

    test_file("mutable", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_for_loop() {
    let private_inputs = r#"{"private_input": ["2", "3", "4"]}"#;
    let public_inputs = r#"{"public_input": ["9"]}"#;

    test_file("for_loop", public_inputs, private_inputs, vec![]);
}

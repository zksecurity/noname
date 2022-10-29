use std::path::Path;

use crate::{
    compiler,
    constants::Field,
    inputs::{parse_inputs, ExtField},
    prover::compile_to_indexes,
    type_checker::Dependencies,
};

fn test_file(
    file_name: &str,
    public_inputs: &str,
    private_inputs: &str,
    expected_public_output: Vec<Field>,
) {
    let version = env!("CARGO_MANIFEST_DIR");
    let prefix = Path::new(version).join("examples");

    // read noname file
    let code = std::fs::read_to_string(prefix.clone().join(format!("{file_name}.no"))).unwrap();

    // compile
    let compiled_circuit = compiler::compile(&code, Dependencies::default()).unwrap();
    let (prover_index, verifier_index) = compile_to_indexes(compiled_circuit).unwrap();

    // check compiled ASM only if it's not too large
    if prover_index.len() < 100 {
        let expected_asm =
            std::fs::read_to_string(prefix.clone().join(format!("{file_name}.asm"))).unwrap();

        let obtained_asm = prover_index.asm(false);
        if obtained_asm != expected_asm {
            eprintln!("obtained:");
            eprintln!("{obtained_asm}");
            eprintln!("expected:");
            eprintln!("{expected_asm}");
            panic!("Obtained ASM does not match expected ASM");
        }
    }

    // parse inputs
    let public_inputs = parse_inputs(public_inputs).unwrap();
    let private_inputs = parse_inputs(private_inputs).unwrap();

    // create proof
    let (proof, full_public_inputs, public_output) = prover_index
        .prove(public_inputs, private_inputs, false)
        .unwrap();

    if public_output != expected_public_output {
        eprintln!("obtained by executing the circuit:");
        public_output.iter().for_each(|x| eprintln!("- {x}"));
        eprintln!("passed as output by the verifier:");
        expected_public_output
            .iter()
            .for_each(|x| eprintln!("- {x}"));
        panic!("Obtained output does not match expected output");
    }

    // verify proof
    verifier_index.verify(full_public_inputs, proof).unwrap();
}

#[test]
fn test_arithmetic() {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

    println!("public inputs: {:?}", public_inputs);
    println!("private inputs: {:?}", private_inputs);

    test_file("arithmetic", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_public_output() {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

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

    let public_inputs = &format!(r#"{{"public_input": "{digest_dec}"}}"#);

    test_file("poseidon", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_bool() {
    let private_inputs = r#"{"private_input": false}"#;
    let public_inputs = r#"{"public_input": true}"#;

    test_file("bool", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_mutable() {
    let private_inputs = r#"{"xx": "2", "yy": "3"}"#;
    let public_inputs = r#"{}"#;

    test_file("mutable", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_for_loop() {
    let private_inputs = r#"{"private_input": ["2", "3", "4"]}"#;
    let public_inputs = r#"{"public_input": "9"}"#;

    test_file("for_loop", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_array() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"public_input": ["1", "2"]}"#;

    test_file("array", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_equals() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": ["3", "3"]}"#;

    test_file("equals", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_types() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1", "yy": "2"}"#;

    test_file("types", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_const() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"player": "1"}"#;
    let expected_public_output = vec![Field::from(2)];

    test_file(
        "const",
        public_inputs,
        private_inputs,
        expected_public_output,
    );
}

#[test]
fn test_functions() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"one": "1"}"#;

    test_file("functions", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_methods() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1"}"#;

    test_file("methods", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_types_array() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1", "yy": "4"}"#;

    test_file("types_array", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_iterate() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"bedroom_holes": "2"}"#;
    let expected_public_output = vec![Field::from(4)];

    test_file(
        "iterate",
        public_inputs,
        private_inputs,
        expected_public_output,
    );
}

#[test]
fn test_assignment() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "2"}"#;

    test_file("assignment", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_if_else() {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1"}"#;

    test_file("if_else", public_inputs, private_inputs, vec![]);
}

#[test]
fn test_sudoku() {
    let private_inputs = r#"{"solution": { "inner": ["9", "5", "3", "6", "2", "1", "7", "8", "4", "1", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "5", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "7", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "7", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "7", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}"#;
    let public_inputs = r#"{"grid": { "inner": ["0", "5", "3", "6", "2", "1", "7", "8", "4", "0", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "0", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "0", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "0", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "0", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}"#;

    test_file("sudoku", public_inputs, private_inputs, vec![]);
}

use std::{path::Path, str::FromStr};

use rstest::rstest;

use crate::{
    backends::{
        kimchi::{KimchiVesta, VestaField},
        r1cs::R1CS,
        BackendKind,
    }, compiler::{compile, typecheck_next_file, Sources}, inputs::{parse_inputs, ExtField}, stdlib::init_stdlib_dep, type_checker::TypeChecker
};

fn test_file(
    file_name: &str,
    public_inputs: &str,
    private_inputs: &str,
    expected_public_output: Vec<&str>,
    backend: BackendKind,
) -> miette::Result<()> {
    let version = env!("CARGO_MANIFEST_DIR");
    let prefix_examples = Path::new(version).join("examples");

    // read noname file
    let code =
        std::fs::read_to_string(prefix_examples.clone().join(format!("{file_name}.no"))).unwrap();

    // parse inputs
    let public_inputs = parse_inputs(public_inputs).unwrap();
    let private_inputs = parse_inputs(private_inputs).unwrap();

    match backend {
        BackendKind::KimchiVesta(kimchi_vesta) => {
            // compile
            let mut sources = Sources::new();
            let mut tast = TypeChecker::new();
            let mut node_id = 0;
            node_id = init_stdlib_dep(&mut sources, &mut tast, node_id);
            let this_module = None;
            let _node_id = typecheck_next_file(
                &mut tast,
                this_module,
                &mut sources,
                file_name.to_string(),
                code.clone(),
                node_id,
            )
            .unwrap();

            let compiled_circuit = compile(&sources, tast, kimchi_vesta)?;

            let (prover_index, verifier_index) = compiled_circuit.compile_to_indexes().unwrap();

            // check compiled ASM only if it's not too large
            let prefix_asm = prefix_examples.join("fixture/asm/kimchi");
            if prover_index.len() < 100 {
                let expected_asm =
                    std::fs::read_to_string(prefix_asm.clone().join(format!("{file_name}.asm")))
                        .unwrap();

                let obtained_asm = prover_index.asm(&mut Sources::new(), false);
                if obtained_asm != expected_asm {
                    eprintln!("obtained:");
                    eprintln!("{obtained_asm}");
                    eprintln!("expected:");
                    eprintln!("{expected_asm}");
                    panic!("Obtained ASM does not match expected ASM");
                }
            }

            // create proof
            let (proof, full_public_inputs, public_output) = prover_index.prove(
                &sources,
                public_inputs.clone(),
                private_inputs.clone(),
                false,
            )?;

            let expected_public_output = expected_public_output
                .iter()
                .map(|x| VestaField::from_str(x).unwrap())
                .collect::<Vec<_>>();

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
        BackendKind::R1csBls12_381(r1cs) => {
            // compile
            let mut sources = Sources::new();
            let mut tast = TypeChecker::new();
            let mut node_id = 0;
            node_id = init_stdlib_dep(&mut sources, &mut tast, node_id);
            let this_module = None;
            let _node_id = typecheck_next_file(
                &mut tast,
                this_module,
                &mut sources,
                file_name.to_string(),
                code.clone(),
                0,
            )
            .unwrap();

            let compiled_circuit = compile(&sources, tast, r1cs)?;

            // this should check the constraints
            let generated_witness = compiled_circuit
                .generate_witness(public_inputs.clone(), private_inputs.clone())
                .unwrap();

            // check the ASM
            if compiled_circuit.circuit.backend.num_constraints() < 100 {
                let prefix_asm = prefix_examples.join("fixture/asm/r1cs");
                let expected_asm =
                    std::fs::read_to_string(prefix_asm.clone().join(format!("{file_name}.asm")))
                        .unwrap();
                let obtained_asm = compiled_circuit.asm(&Sources::new(), false);

                if obtained_asm != expected_asm {
                    eprintln!("obtained:");
                    eprintln!("{obtained_asm}");
                    eprintln!("expected:");
                    eprintln!("{expected_asm}");
                    panic!("Obtained ASM does not match expected ASM");
                }
            }

            let expected_public_output = expected_public_output
                .iter()
                .map(|x| crate::backends::r1cs::R1csBls12381Field::from_str(x).unwrap())
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
        }
        BackendKind::R1csBn254(_) => todo!(),
    }

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_arithmetic(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input": "2"}"#;
    let private_inputs = r#"{"private_input": "2"}"#;

    test_file("arithmetic", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_public_output(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

    test_file(
        "public_output",
        public_inputs,
        private_inputs,
        vec!["8"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_lc_return(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

    test_file(
        "lc_return",
        public_inputs,
        private_inputs,
        vec!["2"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
//todo: #[case::r1cs(BackendKind::R1CS(R1CS::new()))]
fn test_poseidon(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"private_input": ["1", "1"]}"#;
    let private_input = [1.into(), 1.into()];
    let digest = crate::helpers::poseidon(private_input.clone());
    let digest_dec = digest.to_dec_string();
    assert_eq!(
        "3654913405619483358804575553468071097765421484960111776885779739261304758583",
        digest_dec
    );

    let public_inputs = &format!(r#"{{"public_input": "{digest_dec}"}}"#);

    test_file("poseidon", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_bool(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"private_input": false}"#;
    let public_inputs = r#"{"public_input": true}"#;

    test_file("bool", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_mutable(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"xx": "2", "yy": "3"}"#;
    let public_inputs = r#"{}"#;

    test_file("mutable", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_for_loop(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"private_input": ["2", "3", "4"]}"#;
    let public_inputs = r#"{"public_input": "9"}"#;

    test_file("for_loop", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_dup_var(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"private_input": ["1", "2", "2"]}"#;
    let public_inputs = r#"{"public_input": "10"}"#;

    test_file("dup_var", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_array(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"public_input": ["1", "2"]}"#;

    test_file("array", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_equals(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": ["3", "3"]}"#;

    test_file("equals", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_not_equal(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": ["1", "2"]}"#;

    test_file("not_equal", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_types(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1", "yy": "2"}"#;

    test_file("types", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_const(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"player": "1"}"#;
    let expected_public_output = vec!["2"];

    test_file(
        "const",
        public_inputs,
        private_inputs,
        expected_public_output,
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_functions(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"one": "1"}"#;

    test_file("functions", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_methods(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1"}"#;

    test_file("methods", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_types_array(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1", "yy": "4"}"#;

    test_file(
        "types_array",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_iterate(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"bedroom_holes": "2"}"#;
    let expected_public_output = vec!["4"];

    test_file(
        "iterate",
        public_inputs,
        private_inputs,
        expected_public_output,
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_assignment(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "2"}"#;

    test_file("assignment", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_if_else(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1"}"#;

    test_file("if_else", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_sudoku(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{"solution": { "inner": ["9", "5", "3", "6", "2", "1", "7", "8", "4", "1", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "5", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "7", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "7", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "7", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}"#;
    let public_inputs = r#"{"grid": { "inner": ["0", "5", "3", "6", "2", "1", "7", "8", "4", "0", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "0", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "0", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "0", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "0", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}"#;

    test_file("sudoku", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_literals(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"public_input": "42"}"#;

    test_file("literals", public_inputs, private_inputs, vec![], backend)?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_public_output_array(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

    test_file(
        "public_output_array",
        public_inputs,
        private_inputs,
        vec!["8", "2"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_types_array_output(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx": "1", "yy": "4"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "types_array_output",
        public_inputs,
        private_inputs,
        vec!["2", "4", "1", "8"], // 2x, y, x, 2y
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_public_output_bool(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input": "1"}"#;
    let private_inputs = r#"{"private_input": "1"}"#;

    test_file(
        "public_output_bool",
        public_inputs,
        private_inputs,
        vec!["1"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_public_output_types(#[case] backend: BackendKind) -> miette::Result<()> {
    let private_inputs = r#"{}"#;
    let public_inputs = r#"{"xx": "1", "yy": "2"}"#;

    test_file(
        "public_output_types",
        public_inputs,
        private_inputs,
        vec!["1", "2"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_repeated_array(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"public_input":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_repeated_array",
        public_inputs,
        private_inputs,
        vec!["1", "1", "1"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_array_access(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_array_access",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_array_nested(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_array_nested",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_fn_multi_init(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_fn_multi_init",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn generic_method_multi_init(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"1"}"#;
    let private_inputs = r#"{"yy":"2"}"#;

    test_file(
        "generic_method_multi_init",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_for_loop(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_for_loop",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_builtin_bits(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"xx":"2"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_builtin_bits",
        public_inputs,
        private_inputs,
        vec![],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_iterator(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"arr":["1", "2", "3"]}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_iterator",
        public_inputs,
        private_inputs,
        vec!["1"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_nested_func(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"val":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_nested_func",
        public_inputs,
        private_inputs,
        vec!["1", "1", "1"],
        backend,
    )?;

    Ok(())
}

#[rstest]
#[case::kimchi_vesta(BackendKind::KimchiVesta(KimchiVesta::new(false)))]
#[case::r1cs(BackendKind::R1csBls12_381(R1CS::new()))]
fn test_generic_nested_method(#[case] backend: BackendKind) -> miette::Result<()> {
    let public_inputs = r#"{"val":"1"}"#;
    let private_inputs = r#"{}"#;

    test_file(
        "generic_nested_method",
        public_inputs,
        private_inputs,
        vec!["1", "1", "1"],
        backend,
    )?;

    Ok(())
}

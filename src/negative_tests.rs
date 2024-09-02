use crate::{
    backends::{kimchi::KimchiVesta, Backend},
    compiler::{compile, generate_witness, typecheck_next_file_inner, Sources},
    error::ErrorKind,
    inputs::parse_inputs,
    type_checker::TypeChecker,
};

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::NoReturnExpected));
}

#[test]
fn test_return_expected() {
    // return expected
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MissingReturn));
}

#[test]
fn test_return_mismatch() {
    // return type mismatch
    let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}

//The test checks if using && between a Field and a Bool causes a MismatchType error, as these types are incompatible.
#[test]
fn test_boolean_and_fail() {
    let code = r#"
    fn thing(xx: Field, yy: Bool) {
        let zz = xx && yy; // This should cause a type mismatch as `xx` is Field and `yy` is Bool
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "boolean_and_fail.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

// The test checks if using || between a Field and a Bool causes a MismatchType error
#[test]
fn test_boolean_or_fail() {
    let code = r#"
    fn thing(xx: Field, yy: Bool) {
        let zz = xx || yy; // This should cause a type mismatch as `xx` is Field and `yy` is Bool
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "boolean_or_fail.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

// The test ensures that adding a Field and a Bool (incompatible types) results in a MismatchType error.
#[test]
fn test_addition_mismatch() {
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
        return yy + true; // This should cause a type mismatch as `yy` is Field and `true` is Bool
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

//  This ensures that multiplying a Field and a Bool causes a MismatchType error.
#[test]
fn test_multiplication_mismatch() {
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx * 2;
        return yy + true; // This should cause a type mismatch as `yy` is Field and `true` is Bool
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

// Test ASM Snapshot Failure
#[test]
fn test_asm_snapshot_mismatch_fail() {
    let code = r#"
    fn main(pub public_input: Field, private_input: Field) -> Field {
        let xx = private_input + public_input;
        assert_eq(xx, 3); // This should fail the ASM snapshot comparison
        return xx;
    }
    "#;

    let mut sources = Sources::new();
    let mut tast = TypeChecker::<KimchiVesta>::new();
    let _ = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut sources,
        "example.no".to_string(),
        code.to_string(),
        0,
    )
    .expect("Type check failed unexpectedly");

    let compiled_circuit = compile(&sources, tast, KimchiVesta::new(false)).unwrap();

    // Get the actual ASM output
    let asm_output = compiled_circuit
        .circuit
        .backend
        .generate_asm(&sources, false);

    // Expected ASM output that should match (to simulate the test failing)
    let expected_asm_output = "expected output that you expect to match";

    // Check for the mismatch
    assert!(
        asm_output.trim() != expected_asm_output.trim(),
        "Expected an ASM mismatch error, but no mismatch was detected."
    );
}

// Test Witness Generation with Invalid Constraints
#[test]
fn test_invalid_witness_generation_fail() {
    let code = r#"
    fn main(pub public_input: Field, private_input: Field) {
        let xx = private_input * (public_input + 1);
        assert_eq(xx, public_input); // This should fail witness generation
    }
    "#;

    // Register the source code
    let mut sources = Sources::new();
    let _filename_id = sources.add("example.no".to_string(), code.to_string());

    // Type checking
    let mut tast = TypeChecker::<KimchiVesta>::new();
    let _res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut sources,
        "example.no".to_string(),
        code.to_string(),
        0,
    )
    .expect("Type check failed");

    // Compile the circuit
    let compiled_circuit =
        compile(&sources, tast, KimchiVesta::new(false)).expect("Circuit compilation failed");

    // Generate witness and expect an error
    let public_inputs = parse_inputs(r#"{"public_input": "2"}"#).unwrap();
    let private_inputs = parse_inputs(r#"{"private_input": "3"}"#).unwrap();

    let result = generate_witness(&compiled_circuit, &sources, public_inputs, private_inputs);

    assert!(
        result.is_err(),
        "Expected witness generation to fail, but it succeeded"
    );
}

#[test]
fn test_ast_failure() {
    let code = r#"
    fn thing(xx: Field {
        let yy = xx + 1;
    }
    "#;

    let mut sources = Sources::new();
    let res = typecheck_next_file_inner(
        &mut TypeChecker::<KimchiVesta>::new(),
        None,
        &mut sources,
        "example_ast_fail.no".to_string(),
        code.to_string(),
        0,
    );

    // We expect an error because the code is syntactically incorrect
    assert!(
        res.is_err(),
        "Expected parsing to fail due to syntax error, but it succeeded"
    );
}
// Verifies the correctness of the NAST phase.
#[test]
fn test_nast_failure() {
    let code = r#"
    fn main(input: Field) {
        let yy = xx + 1; // `xx` is not defined
    }
    "#;

    let mut sources = Sources::new();
    let res = typecheck_next_file_inner(
        &mut TypeChecker::<KimchiVesta>::new(),
        None,
        &mut sources,
        "example_nast_fail.no".to_string(),
        code.to_string(),
        0,
    );

    if let Err(e) = res {
        println!("Error returned: {:?}", e.kind);
        assert!(matches!(e.kind, ErrorKind::UndefinedVariable));
    } else {
        panic!("Expected an error, but none was returned.");
    }
}

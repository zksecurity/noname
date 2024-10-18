use crate::error::{self, ErrorKind};

use super::{test_stdlib, test_stdlib_code};
use error::Result;
use rstest::rstest;

// code template
static LESS_THAN_TPL: &str = r#"
use std::comparator;
use std::int;

fn main(pub lhs: Field, rhs: Field) -> Bool {
    let lhs_u = int::{}.new(lhs);
    let rhs_u = int::{}.new(rhs);

    let res = lhs_u.less_than(rhs_u);

    return res;
}
"#;

static LESS_THAN_EQ_TPL: &str = r#"
use std::comparator;
use std::int;

fn main(pub lhs: Field, rhs: Field) -> Bool {
    let lhs_u = int::{}.new(lhs);
    let rhs_u = int::{}.new(rhs);

    let res = lhs_u.less_eq_than(rhs_u);

    return res;
}
"#;

#[rstest]
#[case(r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case(r#"{"lhs": "1"}"#, r#"{"rhs": "0"}"#, vec!["0"])]
fn test_less_than(
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    test_stdlib(
        "comparator/less_than/less_than_main.no",
        Some("comparator/less_than/less_than.asm"),
        public_inputs,
        private_inputs,
        expected_output,
    )?;

    Ok(())
}

#[test]
fn test_less_than_witness_failure() -> Result<()> {
    let public_inputs = r#"{"lhs": "4"}"#;
    let private_inputs = r#"{"rhs": "0"}"#;

    let err = test_stdlib(
        "comparator/less_than/less_than_main.no",
        None,
        public_inputs,
        private_inputs,
        vec![],
    )
    .err()
    .expect("expected witness error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

#[rstest]
#[case("Uint8", r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case("Uint16", r#"{"lhs": "1"}"#, r#"{"rhs": "0"}"#, vec!["0"])]
#[case("Uint32", r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case("Uint64", r#"{"lhs": "1"}"#, r#"{"rhs": "0"}"#, vec!["0"])]
fn test_uint_less_than(
    #[case] int_type: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    // Replace placeholders with the given integer type.
    let code = LESS_THAN_TPL.replace("{}", int_type);

    // Call the test function with the given inputs and expected output.
    test_stdlib_code(&code, None, public_inputs, private_inputs, expected_output)?;

    Ok(())
}

#[rstest]
#[case("Uint8", r#"{"lhs": "256"}"#, r#"{"rhs": "0"}"#)] // Uint8 overflow
#[case("Uint16", r#"{"lhs": "65536"}"#, r#"{"rhs": "0"}"#)] // Uint16 overflow
#[case("Uint32", r#"{"lhs": "4294967296"}"#, r#"{"rhs": "0"}"#)] // Uint32 overflow
#[case("Uint64", r#"{"lhs": "18446744073709551616"}"#, r#"{"rhs": "0"}"#)] // Uint64 overflow
fn test_uint_less_than_range_failure(
    #[case] int_type: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
) -> Result<()> {
    let code = LESS_THAN_TPL.replace("{}", int_type);

    // Test that the provided inputs result in an error due to overflow.
    let err = test_stdlib_code(&code, None, public_inputs, private_inputs, vec!["0"])
        .err()
        .expect("expected witness error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

// Test for less than or equal scenarios
#[rstest]
#[case(r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])] // True case (lhs < rhs)
#[case(r#"{"lhs": "1"}"#, r#"{"rhs": "1"}"#, vec!["1"])] // True case (lhs == rhs)
#[case(r#"{"lhs": "1"}"#, r#"{"rhs": "0"}"#, vec!["0"])] // False case
fn test_less_eq_than(
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    test_stdlib(
        "comparator/less_eq_than/less_eq_than_main.no",
        Some("comparator/less_eq_than/less_eq_than.asm"),
        public_inputs,
        private_inputs,
        expected_output,
    )?;

    Ok(())
}

// implement the rest for less than eq

#[rstest]
#[case("Uint8", r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case("Uint16", r#"{"lhs": "1"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case("Uint32", r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#, vec!["1"])]
#[case("Uint64", r#"{"lhs": "1"}"#, r#"{"rhs": "0"}"#, vec!["0"])]
fn test_uint_less_eq_than(
    #[case] int_type: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    let code = LESS_THAN_EQ_TPL.replace("{}", int_type);

    test_stdlib_code(&code, None, public_inputs, private_inputs, expected_output)?;

    Ok(())
}

#[rstest]
#[case("Uint8", r#"{"lhs": "256"}"#, r#"{"rhs": "0"}"#)] // Uint8 overflow
#[case("Uint16", r#"{"lhs": "65536"}"#, r#"{"rhs": "0"}"#)] // Uint16 overflow
#[case("Uint32", r#"{"lhs": "4294967296"}"#, r#"{"rhs": "0"}"#)] // Uint32 overflow
#[case("Uint64", r#"{"lhs": "18446744073709551616"}"#, r#"{"rhs": "0"}"#)] // Uint64 overflow
fn test_uint_less_eq_than_range_failure(
    #[case] int_type: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
) -> Result<()> {
    let code = LESS_THAN_EQ_TPL.replace("{}", int_type);

    let err = test_stdlib_code(&code, None, public_inputs, private_inputs, vec!["0"])
        .err()
        .expect("expected witness error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

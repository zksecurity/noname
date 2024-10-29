use crate::error::{self, ErrorKind};

use super::test_stdlib_code;
use error::Result;
use rstest::rstest;

// code template
static TPL: &str = r#"
use std::int;

fn main(pub lhs: Field, rhs: Field) -> Field {
    let lhs_u = int::{inttyp}.new(lhs);
    let rhs_u = int::{inttyp}.new(rhs);

    let res = lhs_u.{opr}(rhs_u);

    return res.inner;
}
"#;

#[rstest]
#[case("Uint8", "add", r#"{"lhs": "2"}"#, r#"{"rhs": "2"}"#, vec!["4"])]
#[case("Uint8", "sub", r#"{"lhs": "2"}"#, r#"{"rhs": "2"}"#, vec!["0"])]
#[case("Uint8", "mul", r#"{"lhs": "2"}"#, r#"{"rhs": "2"}"#, vec!["4"])]
#[case("Uint8", "div", r#"{"lhs": "5"}"#, r#"{"rhs": "3"}"#, vec!["1"])]
#[case("Uint8", "mod", r#"{"lhs": "5"}"#, r#"{"rhs": "3"}"#, vec!["2"])]
fn test_uint_ops(
    #[case] int_type: &str,
    #[case] operation: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    // Replace placeholders with the given integer type.
    let code = TPL
        .replace("{inttyp}", int_type)
        .replace("{opr}", operation);

    // Call the test function with the given inputs and expected output.
    test_stdlib_code(&code, None, public_inputs, private_inputs, expected_output)?;

    Ok(())
}

/// test overflow after operation
#[rstest]
#[case("Uint8", "add", r#"{"lhs": "255"}"#, r#"{"rhs": "1"}"#)]
#[case("Uint8", "sub", r#"{"lhs": "0"}"#, r#"{"rhs": "1"}"#)]
#[case("Uint8", "mul", r#"{"lhs": "255"}"#, r#"{"rhs": "2"}"#)]
fn test_uint_overflow(
    #[case] int_type: &str,
    #[case] operation: &str,
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
) -> Result<()> {
    let code = TPL
        .replace("{inttyp}", int_type)
        .replace("{opr}", operation);

    let err = test_stdlib_code(&code, None, public_inputs, private_inputs, vec![])
        .err()
        .expect("expected overflow error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

use crate::error;

use super::test_stdlib;
use error::Result;

#[test]
fn test_less_than_true() -> Result<()> {
    let public_inputs = r#"{"lhs": "0"}"#;
    let private_inputs = r#"{"rhs": "1"}"#;

    test_stdlib(
        "comparator/less_than/less_than_main.no",
        "comparator/less_than/less_than.asm",
        public_inputs,
        private_inputs,
        vec!["1"],
    )?;

    Ok(())
}

// test false
#[test]
fn test_less_than_false() -> Result<()> {
    let public_inputs = r#"{"lhs": "1"}"#;
    let private_inputs = r#"{"rhs": "0"}"#;

    test_stdlib(
        "comparator/less_than/less_than_main.no",
        "comparator/less_than/less_than.asm",
        public_inputs,
        private_inputs,
        vec!["0"],
    )?;

    Ok(())
}

#[test]
fn test_less_eq_than_true_1() -> Result<()> {
    let public_inputs = r#"{"lhs": "0"}"#;
    let private_inputs = r#"{"rhs": "1"}"#;

    test_stdlib(
        "comparator/less_eq_than/less_eq_than_main.no",
        "comparator/less_eq_than/less_eq_than.asm",
        public_inputs,
        private_inputs,
        vec!["1"],
    )?;

    Ok(())
}

#[test]
fn test_less_eq_than_true_2() -> Result<()> {
    let public_inputs = r#"{"lhs": "1"}"#;
    let private_inputs = r#"{"rhs": "1"}"#;

    test_stdlib(
        "comparator/less_eq_than/less_eq_than_main.no",
        "comparator/less_eq_than/less_eq_than.asm",
        public_inputs,
        private_inputs,
        vec!["1"],
    )?;

    Ok(())
}

#[test]
fn test_less_eq_than_false() -> Result<()> {
    let public_inputs = r#"{"lhs": "1"}"#;
    let private_inputs = r#"{"rhs": "0"}"#;

    test_stdlib(
        "comparator/less_eq_than/less_eq_than_main.no",
        "comparator/less_eq_than/less_eq_than.asm",
        public_inputs,
        private_inputs,
        vec!["0"],
    )?;

    Ok(())
}

// test value overflow modulus
// it shouldn't need user to enter the bit length
// should have a way to restrict and type check the value to a certain bit length

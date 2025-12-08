use crate::error::{self, ErrorKind};

use super::test_stdlib;
use error::Result;
use rstest::rstest;

#[rstest]
#[case(r#"{"val": "0"}"#, vec!["0"])]
#[case(r#"{"val": "1"}"#, vec!["1"])]
fn test_bit_checked(#[case] public_inputs: &str, #[case] expected_output: Vec<&str>) -> Result<()> {
    test_stdlib(
        "bits/bit_checked.no",
        None,
        public_inputs,
        r#"{}"#,
        expected_output,
    )?;

    Ok(())
}

#[test]
fn test_bit_checked_witness_failure() -> Result<()> {
    let public_inputs = r#"{"val": "2"}"#; // should break range check

    let err = test_stdlib("bits/bit_checked.no", None, public_inputs, r#"{}"#, vec![])
        .err()
        .expect("expected witness error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

#[rstest]
#[case(r#"{"val": "0"}"#, vec!["0"])] // 0 => false
#[case(r#"{"val": "1"}"#, vec!["1"])] // 1 => true
#[case(r#"{"val": "2"}"#, vec!["0"])] // _ => false
#[case(r#"{"val": "99"}"#, vec!["0"])]
fn test_bit_unchecked(
    #[case] public_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    test_stdlib(
        "bits/bit_unchecked.no",
        None,
        public_inputs,
        r#"{}"#,
        expected_output,
    )?;

    Ok(())
}

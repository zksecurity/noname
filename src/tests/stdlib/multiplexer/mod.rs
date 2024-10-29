use crate::error::{self};

use super::test_stdlib;
use error::Result;
use rstest::rstest;

#[rstest]
#[case(r#"{"xx": [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]}"#, r#"{"sel": "1"}"#, vec!["3", "4", "5"])]
fn test_in_range(
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    test_stdlib(
        "multiplexer/select_element/main.no",
        Some("multiplexer/select_element/main.asm"),
        public_inputs,
        private_inputs,
        expected_output,
    )?;

    Ok(())
}

// require the select idx to be in range
#[rstest]
#[case(r#"{"xx": [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]}"#, r#"{"sel": "3"}"#, vec![])]
fn test_out_range(
    #[case] public_inputs: &str,
    #[case] private_inputs: &str,
    #[case] expected_output: Vec<&str>,
) -> Result<()> {
    use crate::error::ErrorKind;

    let err = test_stdlib(
        "multiplexer/select_element/main.no",
        Some("multiplexer/select_element/main.asm"),
        public_inputs,
        private_inputs,
        expected_output,
    )
    .err()
    .expect("Expected error");

    assert!(matches!(err.kind, ErrorKind::InvalidWitness(..)));

    Ok(())
}

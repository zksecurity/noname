//! Used to parse public and private inputs to a program.

use std::{collections::HashMap, str::FromStr};

use ark_ff::{One, PrimeField, Zero};
use miette::Diagnostic;
use num_bigint::BigUint;
use thiserror::Error;

use crate::{constants::Field, parser::TyKind};

//
// Errors
//

#[derive(Error, Diagnostic, Debug)]
pub enum ParsingError {
    #[error(transparent)]
    IoError(#[from] serde_json::Error),

    #[error("error parsing input {0}")]
    Inputs(String),

    #[error("couldn't convert given field element `{0}`")]
    InvalidField(String),
}

//
// JSON deserialization of top-level hashmap
// (arguments to more stuff)
//

/// An input is a name, and a list of field elements (in decimal).
#[derive(Default, serde::Deserialize)]
pub struct JsonInputs(pub HashMap<String, serde_json::Value>);

pub fn parse_inputs(s: &str) -> Result<JsonInputs, ParsingError> {
    let json_inputs: JsonInputs = serde_json::from_str(s)?;
    Ok(json_inputs)
}

//
// JSON deserialization of a single input
//

pub fn parse_single_input(
    input: serde_json::Value,
    expected_input: &TyKind,
) -> Result<Vec<Field>, ParsingError> {
    use serde_json::Value;

    match (expected_input, input) {
        (TyKind::BigInt, _) => unreachable!(),
        (TyKind::Field, Value::String(ss)) => {
            let cell_value = Field::from_str(&ss).map_err(|_| ParsingError::InvalidField(ss))?;
            return Ok(vec![cell_value]);
        }
        (TyKind::Bool, Value::Bool(bb)) => {
            let ff = if bb { Field::one() } else { Field::zero() };
            return Ok(vec![ff]);
        }

        (TyKind::Array(el_typ, size), Value::Array(values)) => {
            if values.len() != (*size as usize) {
                panic!("wrong size of array");
            }
            let mut res = vec![];
            for value in values {
                let el = parse_single_input(value, el_typ)?;
                res.extend(el);
            }

            return Ok(res);
        }
        (TyKind::Custom(_), _) => todo!(),
        (expected, observed) => {
            dbg!(expected);
            dbg!(observed);
            panic!("mismatch")
        }
    }
}

//
// Helpers
//

pub trait ExtField: PrimeField {
    fn to_dec_string(&self) -> String;
}

impl ExtField for Field {
    fn to_dec_string(&self) -> String {
        let biguint: BigUint = self.into_repr().into();
        biguint.to_str_radix(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extfield() {
        let field = Field::from(42);
        assert_eq!(field.to_dec_string(), "42");
    }
}

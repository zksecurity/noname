//! Used to parse public and private inputs to a program.

use std::{collections::HashMap, str::FromStr};

use ark_ff::PrimeField;
use miette::Diagnostic;
use num_bigint::BigUint;
use thiserror::Error;

use crate::{constants::Field, var::CellValues};

#[derive(Error, Diagnostic, Debug)]
pub enum ParsingError {
    #[error(transparent)]
    IoError(#[from] serde_json::Error),

    #[error("error parsing input {0}")]
    Inputs(String),
}

/// An input is a name, and a list of field elements (in decimal).
#[derive(serde::Deserialize)]
struct JsonInputs(HashMap<String, Vec<String>>);

#[derive(Default, Debug, Clone)]
pub struct Inputs(pub HashMap<String, CellValues>);

pub fn parse_inputs(s: &str) -> Result<Inputs, ParsingError> {
    let json_inputs: JsonInputs = serde_json::from_str(s)?;

    let mut res = HashMap::new();
    for (key, str_values) in json_inputs.0 {
        let mut values = vec![];
        for str_value in str_values {
            let cell_value =
                Field::from_str(&str_value).map_err(|_| ParsingError::Inputs(key.clone()))?;
            values.push(cell_value);
        }
        res.insert(key, CellValues { values });
    }

    Ok(Inputs(res))
}

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

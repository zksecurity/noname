//! Used to parse public and private inputs to a program.

use std::{collections::HashMap, fs::File, io::Read, str::FromStr};

use ark_ff::{One, PrimeField, Zero};
use miette::Diagnostic;
use num_bigint::BigUint;
use thiserror::Error;

use crate::{
    backends::{kimchi::VestaField, Backend},
    parser::types::TyKind,
    type_checker::FullyQualified,
    witness::CompiledCircuit,
};

//
// Errors
//

#[derive(Error, Diagnostic, Debug)]
pub enum ParsingError {
    #[error("JSON parsing error in file {file}: {source}")]
    JsonFileError {
        source: serde_json::Error,
        file: String,
    },

    #[error("error parsing input {0}")]
    Inputs(String),

    #[error("couldn't convert given field element `{0}`")]
    InvalidField(String),

    #[error("mismatch between expected argument format ({0}), and given argument in JSON (`{1}`)")]
    MismatchJsonArgument(TyKind, serde_json::Value),
    #[error("Incorrect array size: Expected {0}, Got {1})")]
    ArraySizeMismatch(usize, usize),
    #[error("Incorrect struct fields count. {0} expected {1}, Got {2})")]
    StructFieldCountMismatch(String, usize, usize),
    #[error("Compiler bug: failed to locate struct name `{0}` given as input")]
    MissingStructIdent(String),
    #[error("Failed to locate struct field name `{0}` in JSON input")]
    MissingStructFieldIdent(String),
}

//
// JSON deserialization of top-level hashmap
// (arguments to more stuff)
//

/// An input is a name, and a list of field elements (in decimal).
#[derive(Default, serde::Deserialize, Clone)]
pub struct JsonInputs(pub HashMap<String, serde_json::Value>);

pub fn parse_inputs(s: &str) -> Result<JsonInputs, ParsingError> {
    if let Ok(json_inputs) = serde_json::from_str(s) {
        return Ok(json_inputs);
    }

    let mut file = File::open(s).map_err(|_| ParsingError::Inputs(s.to_string()))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|_| ParsingError::Inputs(s.to_string()))?;

    let json_inputs = serde_json::from_str(&contents).map_err(|e| ParsingError::JsonFileError {
        source: e,
        file: s.to_string(),
    })?;

    Ok(json_inputs)
}

//
// JSON deserialization of a single input
//

impl<B: Backend> CompiledCircuit<B> {
    pub fn parse_single_input(
        &self,
        input: serde_json::Value,
        expected_input: &TyKind,
    ) -> Result<Vec<B::Field>, ParsingError> {
        use serde_json::Value;

        match (expected_input, input) {
            (TyKind::Field { constant: true }, _) => unreachable!(),
            (TyKind::Field { constant: false }, Value::String(ss)) => {
                let cell_value =
                    B::Field::from_str(&ss).map_err(|_| ParsingError::InvalidField(ss))?;
                Ok(vec![cell_value])
            }
            (TyKind::Bool, Value::Bool(bb)) => {
                let ff = if bb {
                    B::Field::one()
                } else {
                    B::Field::zero()
                };
                Ok(vec![ff])
            }

            (TyKind::Array(el_typ, size), Value::Array(values)) => {
                if values.len() != (*size as usize) {
                    Err(ParsingError::ArraySizeMismatch(
                        values.len(),
                        *size as usize,
                    ))?
                }
                let mut res = vec![];
                for value in values {
                    let el = self.parse_single_input(value, el_typ)?;
                    res.extend(el);
                }

                Ok(res)
            }
            (
                TyKind::Custom {
                    module,
                    name: struct_name,
                },
                Value::Object(mut map),
            ) => {
                // get fields of struct
                let qualified = FullyQualified::new(module, struct_name);
                let struct_info = self
                    .circuit
                    .struct_info(&qualified)
                    .ok_or_else(|| ParsingError::MissingStructIdent(struct_name.to_string()))?;
                let fields = &struct_info.fields;

                // make sure that they're the same length
                if fields.len() != map.len() {
                    Err(ParsingError::StructFieldCountMismatch(
                        struct_info.name.to_string(),
                        map.len(),
                        fields.len(),
                    ))?;
                }

                // parse each field
                let mut res = vec![];
                for (field_name, field_ty, _attribute) in fields {
                    let value = map.remove(field_name).ok_or_else(|| {
                        ParsingError::MissingStructFieldIdent(field_name.to_string())
                    })?;
                    let parsed = self.parse_single_input(value, field_ty)?;
                    res.extend(parsed);
                }

                Ok(res)
            }
            (expected, observed) => {
                return Err(ParsingError::MismatchJsonArgument(
                    expected.clone(),
                    observed,
                ));
            }
        }
    }
}

//
// Helpers
//

pub trait ExtField /* : PrimeField*/ {
    fn to_dec_string(&self) -> String;
}

impl ExtField for VestaField {
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
        let field = VestaField::from(42);
        assert_eq!(field.to_dec_string(), "42");
    }
}

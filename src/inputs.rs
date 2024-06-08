//! Used to parse public and private inputs to a program.

use std::{collections::HashMap, fs, str::FromStr};
use camino::Utf8PathBuf as PathBuf;
use ark_ff::{One, PrimeField, Zero};
use miette::Diagnostic;
use num_bigint::BigUint;
use thiserror::Error;

use crate::{
    backends::{kimchi::VestaField, Backend}, error::Error, parser::types::TyKind, type_checker::FullyQualified, witness::CompiledCircuit
};

//
// Errors
//

#[derive(Error, Diagnostic, Debug)]
pub enum ParsingError {
    #[error(transparent)]
    IoError(#[from] serde_json::Error),

    #[error("error parsing JSON data `{0}`")]
    JsonParsingError(#[from] std::io::Error),

    #[error("error parsing input {0}")]
    Inputs(String),

    #[error("couldn't convert given field element `{0}`")]
    InvalidField(String),

    #[error("mismatch between expected argument format ({0}), and given argument in JSON (`{1}`)")]
    MismatchJsonArgument(TyKind, serde_json::Value),

    #[error("path does not exist: {0}")]
    PathDoesNotExist(String),

    #[error("path is a directory, not a file: {0}")]
    PathIsDirectory(String),
}

//
// JSON deserialization of top-level hashmap
// (arguments to more stuff)
//

/// An input is a name, and a list of field elements (in decimal).
#[derive(Default, serde::Deserialize, Clone)]
pub struct JsonInputs(pub HashMap<String, serde_json::Value>);

// pub struct 

pub fn parse_inputs(s: &str) -> Result<JsonInputs, ParsingError> {
    let json_inputs: JsonInputs = serde_json::from_str(s)?;
    Ok(json_inputs)
}

pub fn parse_json_file_inputs(s: &str) -> Result<JsonInputs, ParsingError> {
   let path = PathBuf::from(s); 
   
   if !path.exists() {
     return Err(ParsingError::PathDoesNotExist(path.into_string()));
   };

   if path.is_dir() {
     return Err(ParsingError::PathIsDirectory(path.into_string()));
   };

   let json_file_content = fs::read_to_string(&path)
   
   .map_err(ParsingError::JsonParsingError)?;

   let json_inputs = parse_inputs(&json_file_content)?;

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
            (TyKind::BigInt, _) => unreachable!(),
            (TyKind::Field, Value::String(ss)) => {
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
                    panic!("wrong size of array");
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
                    .expect("compiler bug: couldn't find struct given as input");
                let fields = &struct_info.fields;

                // make sure that they're the same length
                if fields.len() != map.len() {
                    panic!("wrong number of fields in struct (TODO: better error)");
                }

                // parse each field
                let mut res = vec![];
                for (field_name, field_ty) in fields {
                    let value = map
                    .remove(field_name)
                    .ok_or_else(|| {
                        format!("couldn't find field `{field_name}` in given JSON input (TODO: better error)")
                    })
                    .unwrap();
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

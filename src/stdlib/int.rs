use std::vec;

use kimchi::o1_utils::FieldHelpers;

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    parser::types::GenericParameters,
    var::{ConstOrCell, Value, Var},
};

use super::{FnInfoType, Module};

const DIVMOD_FN: &str = "divmod(dividend: Field, divisor: Field) -> [Field; 2]";

pub struct IntLib {}

impl Module for IntLib {
    const MODULE: &'static str = "int";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![(DIVMOD_FN, divmod_fn, false)]
    }
}

/// Divides two field elements and returns the quotient and remainder.
fn divmod_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get two vars
    let dividend_info = &vars[0];
    let divisor_info = &vars[1];

    // retrieve the values
    let dividend_var = &dividend_info.var[0];
    let divisor_var = &divisor_info.var[0];

    match (dividend_var, divisor_var) {
        // two constants
        (ConstOrCell::Const(a), ConstOrCell::Const(b)) => {
            // convert to bigints
            let a = a.to_biguint();
            let b = b.to_biguint();

            let quotient = a.clone() / b.clone();
            let remainder = a % b;

            // convert back to fields
            let quotient = B::Field::from_biguint(&quotient).unwrap();
            let remainder = B::Field::from_biguint(&remainder).unwrap();

            Ok(Some(Var::new(
                vec![ConstOrCell::Const(quotient), ConstOrCell::Const(remainder)],
                span,
            )))
        }

        _ => {
            let quotient = compiler
                .backend
                .new_internal_var(Value::Div(dividend_var.clone(), divisor_var.clone()), span);
            let remainder = compiler
                .backend
                .new_internal_var(Value::Mod(dividend_var.clone(), divisor_var.clone()), span);

            Ok(Some(Var::new(
                vec![ConstOrCell::Cell(quotient), ConstOrCell::Cell(remainder)],
                span,
            )))
        }
    }
}

//! Builtins are imported by default.

use ark_ff::One;

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::types::{GenericParameters, TyKind},
    var::{ConstOrCell, Var},
};

use super::{FnInfoType, Module};

pub const QUALIFIED_BUILTINS: &str = "std/builtins";
pub const BUILTIN_FN_NAMES: [&str; 2] = ["assert", "assert_eq"];

const ASSERT_FN: &str = "assert(condition: Bool)";
const ASSERT_EQ_FN: &str = "assert_eq(lhs: Field, rhs: Field)";

pub struct BuiltinsLib {}

impl Module for BuiltinsLib {
    const MODULE: &'static str = "builtins";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>)> {
        vec![(ASSERT_FN, assert_fn), (ASSERT_EQ_FN, assert_eq_fn)]
    }
}

/// Asserts that two vars are equal.
fn assert_eq_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get two vars
    assert_eq!(vars.len(), 2);
    let lhs_info = &vars[0];
    let rhs_info = &vars[1];

    // they are both of type field
    if !matches!(lhs_info.typ, Some(TyKind::Field | TyKind::BigInt)) {
        panic!(
            "the lhs of assert_eq must be of type Field or BigInt. It was of type {:?}",
            lhs_info.typ
        );
    }

    if !matches!(rhs_info.typ, Some(TyKind::Field | TyKind::BigInt)) {
        panic!(
            "the rhs of assert_eq must be of type Field or BigInt. It was of type {:?}",
            rhs_info.typ
        );
    }

    // retrieve the values
    let lhs_var = &lhs_info.var;
    assert_eq!(lhs_var.len(), 1);
    let lhs_cvar = &lhs_var[0];

    let rhs_var = &rhs_info.var;
    assert_eq!(rhs_var.len(), 1);
    let rhs_cvar = &rhs_var[0];

    match (lhs_cvar, rhs_cvar) {
        // two constants
        (ConstOrCell::Const(a), ConstOrCell::Const(b)) => {
            if a != b {
                return Err(Error::new(
                    "constraint-generation",
                    ErrorKind::AssertionFailed,
                    span,
                ));
            }
        }

        // a const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            compiler.backend.assert_eq_const(cvar, *cst, span)
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            compiler.backend.assert_eq_var(lhs, rhs, span)
        }
    }

    Ok(None)
}

/// Asserts that a condition is true.
fn assert_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get a single var
    assert_eq!(vars.len(), 1);

    // of type bool
    let var_info = &vars[0];
    assert!(matches!(var_info.typ, Some(TyKind::Bool)));

    // of only one field element
    let var = &var_info.var;
    assert_eq!(var.len(), 1);
    let cond = &var[0];

    match cond {
        ConstOrCell::Const(cst) => {
            assert!(cst.is_one());
        }
        ConstOrCell::Cell(cvar) => {
            let one = B::Field::one();
            compiler.backend.assert_eq_const(cvar, one, span);
        }
    }

    Ok(None)
}

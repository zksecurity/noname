//! Builtins are imported by default.

use ark_ff::One;
use kimchi::o1_utils::FieldHelpers;
use num_traits::{ToPrimitive, Zero};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    parser::types::{GenericParameters, TyKind},
    var::{ConstOrCell, Value, Var},
};

use super::{FnInfoType, Module};

pub const QUALIFIED_BUILTINS: &str = "std/builtins";
pub const BUILTIN_FN_NAMES: [&str; 2] = ["assert", "assert_eq"];

const ASSERT_FN: &str = "assert(condition: Bool)";
const ASSERT_EQ_FN: &str = "assert_eq(lhs: Field, rhs: Field)";
// todo: currently only supports a single field var
// to support all the types, we can bypass the type check for this log function for now
const LOG_FN: &str = "log(var: Field)";
const DIV_EQ_FN: &str = "div(lhs: Field, rhs: Field) -> Field";
const MOD_EQ_FN: &str = "mod(lhs: Field, rhs: Field) -> Field";
const POW_EQ_FN: &str = "pow(base: Field, exp: Field) -> Field";

pub struct BuiltinsLib {}

impl Module for BuiltinsLib {
    const MODULE: &'static str = "builtins";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>)> {
        vec![
            (ASSERT_FN, assert_fn),
            (ASSERT_EQ_FN, assert_eq_fn),
            (LOG_FN, log_fn),
            (DIV_EQ_FN, div_fn),
            (MOD_EQ_FN, mod_fn),
            (POW_EQ_FN, pow_fn),
        ]
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
    if !matches!(lhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the lhs of assert_eq must be of type Field. It was of type {:?}",
            lhs_info.typ
        );
    }

    if !matches!(rhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the rhs of assert_eq must be of type Field. It was of type {:?}",
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

/// Unconstrained division.
fn div_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get two vars
    assert_eq!(vars.len(), 2);
    let lhs_info = &vars[0];
    let rhs_info = &vars[1];

    // they are both of type field
    if !matches!(lhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the lhs of div must be of type Field. It was of type {:?}",
            lhs_info.typ
        );
    }

    if !matches!(rhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the rhs of div must be of type Field. It was of type {:?}",
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
            if b.is_zero() {
                return Err(Error::new(
                    "constraint-generation",
                    ErrorKind::DivisionByZero,
                    span,
                ));
            }
            let res = *a / b;
            Ok(Some(Var::new(vec![ConstOrCell::Const(res)], span)))
        }

        // a const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar)) => {
            let val = Value::CstDivVar(*cst, cvar.clone());
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
        (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            let val = Value::VarDivCst(cvar.clone(), *cst);
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            let val = Value::VarDivVar(lhs.clone(), rhs.clone());
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
    }
}

/// Logging
fn log_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    println!("---log span: {:?}---", span);
    for var in vars {
        // typ
        println!("typ: {:?}", var.typ);
        // mutable
        println!("mutable: {:?}", var.mutable);
        // var
        var.var.iter().for_each(|v| match v {
            ConstOrCell::Const(cst) => {
                println!("cst: {:?}", cst.pretty());
            }
            ConstOrCell::Cell(cvar) => {
                println!("cvar: {:?}", cvar);
            }
        });
    }

    Ok(None)
}

/// Unconstrained modulo.
fn mod_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get two vars
    assert_eq!(vars.len(), 2);
    let lhs_info = &vars[0];
    let rhs_info = &vars[1];

    // they are both of type field
    if !matches!(lhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the lhs of mod must be of type Field. It was of type {:?}",
            lhs_info.typ
        );
    }

    if !matches!(rhs_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the rhs of mod must be of type Field. It was of type {:?}",
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
            if b.is_zero() {
                return Err(Error::new(
                    "constraint-generation",
                    ErrorKind::DivisionByZero,
                    span,
                ));
            }
            // convert to bigint
            let a = a.to_biguint();
            let b = b.to_biguint();
            let res = a % b;
            Ok(Some(Var::new(
                vec![ConstOrCell::Const(B::Field::from(res))],
                span,
            )))
        }
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar)) => {
            let val = Value::CstModVar(*cst, cvar.clone());
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
        (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            let val = Value::VarModCst(cvar.clone(), *cst);
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            let val = Value::VarModVar(lhs.clone(), rhs.clone());
            let res = compiler.backend.new_internal_var(val, span);
            Ok(Some(Var::new(vec![ConstOrCell::Cell(res)], span)))
        }
    }
}

/// Unconstrained exponentiation.
fn pow_fn<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get two vars
    assert_eq!(vars.len(), 2);
    let base_info = &vars[0];
    let exp_info = &vars[1];

    // they are both of type field
    if !matches!(base_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the base of pow must be of type Field. It was of type {:?}",
            base_info.typ
        );
    }

    if !matches!(exp_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the exp of pow must be of type Field. It was of type {:?}",
            exp_info.typ
        );
    }

    // retrieve the values
    let base_var = &base_info.var;
    assert_eq!(base_var.len(), 1);
    let base_cvar = &base_var[0];

    let exp_var = &exp_info.var;
    assert_eq!(exp_var.len(), 1);
    let exp_cvar = &exp_var[0];

    match (base_cvar, exp_cvar) {
        // two constants
        (ConstOrCell::Const(a), ConstOrCell::Const(b)) => {
            // convert to bigint
            let a = a.to_biguint();
            let b = b.to_biguint();
            let res = a.pow(b.to_u32().expect("expects u32 number"));
            Ok(Some(Var::new(
                vec![ConstOrCell::Const(B::Field::from(res))],
                span,
            )))
        }
        _ => {
            todo!()
        }
    }
}

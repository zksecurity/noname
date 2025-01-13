//! Builtins are imported by default.

use std::sync::Arc;

use ark_ff::{One, Zero};
use kimchi::o1_utils::FieldHelpers;
use num_bigint::BigUint;
use regex::Regex;

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    parser::types::{GenericParameters, ModulePath, TyKind},
    type_checker::FullyQualified,
    var::{ConstOrCell, Value, Var},
};

use super::{FnInfoType, Module};

pub const QUALIFIED_BUILTINS: &str = "std/builtins";
pub const BUILTIN_FN_NAMES: [&str; 3] = ["assert", "assert_eq", "log"];

pub struct BuiltinsLib {}

impl Module for BuiltinsLib {
    const MODULE: &'static str = "builtins";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![
            (AssertFn::SIGNATURE, AssertFn::builtin, false),
            (AssertEqFn::SIGNATURE, AssertEqFn::builtin, true),
            // true -> skip argument type checking for log
            (LogFn::SIGNATURE, LogFn::builtin, true),
        ]
    }
}

/// Represents a comparison that needs to be made
enum Comparison<B: Backend> {
    /// Compare two variables
    Vars(B::Var, B::Var),
    /// Compare a variable with a constant
    VarConst(B::Var, B::Field),
    /// Compare two constants
    Constants(B::Field, B::Field),
}

/// Helper function to generate all comparisons
fn assert_eq_values<B: Backend>(
    compiler: &CircuitWriter<B>,
    lhs_info: &VarInfo<B::Field, B::Var>,
    rhs_info: &VarInfo<B::Field, B::Var>,
    typ: &TyKind,
    span: Span,
) -> Vec<Comparison<B>> {
    let mut comparisons = Vec::new();

    match typ {
        // Field and Bool has the same logic
        TyKind::Field { .. } | TyKind::Bool | TyKind::String(..) => {
            let lhs_var = &lhs_info.var[0];
            let rhs_var = &rhs_info.var[0];
            match (lhs_var, rhs_var) {
                (ConstOrCell::Const(a), ConstOrCell::Const(b)) => {
                    comparisons.push(Comparison::Constants(a.clone(), b.clone()));
                }
                (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
                | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
                    comparisons.push(Comparison::VarConst(cvar.clone(), cst.clone()));
                }
                (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
                    comparisons.push(Comparison::Vars(lhs.clone(), rhs.clone()));
                }
            }
        }

        // Arrays (fixed size)
        TyKind::Array(element_type, size) => {
            let size = *size as usize;
            let element_size = compiler.size_of(element_type);

            // compare each element recursively
            for i in 0..size {
                let start = i * element_size;
                let mut element_comparisons = assert_eq_values(
                    compiler,
                    &VarInfo::new(
                        Var::new(lhs_info.var.range(start, element_size).to_vec(), span),
                        false,
                        Some(*element_type.clone()),
                    ),
                    &VarInfo::new(
                        Var::new(rhs_info.var.range(start, element_size).to_vec(), span),
                        false,
                        Some(*element_type.clone()),
                    ),
                    element_type,
                    span,
                );
                comparisons.append(&mut element_comparisons);
            }
        }

        // Custom types (structs)
        TyKind::Custom { module, name } => {
            let qualified = FullyQualified::new(module, name);
            let struct_info = compiler.struct_info(&qualified).expect("struct not found");

            // compare each field recursively
            let mut offset = 0;
            for (_, field_type) in &struct_info.fields {
                let field_size = compiler.size_of(field_type);
                let mut field_comparisons = assert_eq_values(
                    compiler,
                    &VarInfo::new(
                        Var::new(lhs_info.var.range(offset, field_size).to_vec(), span),
                        false,
                        Some(field_type.clone()),
                    ),
                    &VarInfo::new(
                        Var::new(rhs_info.var.range(offset, field_size).to_vec(), span),
                        false,
                        Some(field_type.clone()),
                    ),
                    field_type,
                    span,
                );
                comparisons.append(&mut field_comparisons);
                offset += field_size;
            }
        }

        // GenericSizedArray should be monomorphized to Array before reaching here
        // no need to handle it seperately
        TyKind::GenericSizedArray(_, _) => {
            unreachable!("GenericSizedArray should be monomorphized")
        }
    }

    comparisons
}

pub trait Builtin {
    const SIGNATURE: &'static str;

    fn builtin<B: Backend>(
        compiler: &mut CircuitWriter<B>,
        generics: &GenericParameters,
        vars: &[VarInfo<B::Field, B::Var>],
        span: Span,
    ) -> Result<Option<Var<B::Field, B::Var>>>;
}

struct AssertEqFn {}
struct AssertFn {}
struct LogFn {}

impl Builtin for AssertEqFn {
    const SIGNATURE: &'static str = "assert_eq(lhs: Field, rhs: Field)";

    /// Asserts that two vars are equal.
    fn builtin<B: Backend>(
        compiler: &mut CircuitWriter<B>,
        _generics: &GenericParameters,
        vars: &[VarInfo<B::Field, B::Var>],
        span: Span,
    ) -> Result<Option<Var<B::Field, B::Var>>> {
        // we get two vars
        assert_eq!(vars.len(), 2);
        let lhs_info = &vars[0];
        let rhs_info = &vars[1];

        // get types of both arguments
        let lhs_type = lhs_info.typ.as_ref().ok_or_else(|| {
            Error::new(
                "constraint-generation",
                ErrorKind::UnexpectedError("No type info for lhs of assertion"),
                span,
            )
        })?;

        let rhs_type = rhs_info.typ.as_ref().ok_or_else(|| {
            Error::new(
                "constraint-generation",
                ErrorKind::UnexpectedError("No type info for rhs of assertion"),
                span,
            )
        })?;

        // they have the same type
        if !lhs_type.match_expected(rhs_type, false) {
            return Err(Error::new(
                "constraint-generation",
                ErrorKind::AssertEqTypeMismatch(lhs_type.clone(), rhs_type.clone()),
                span,
            ));
        }

        // first collect all comparisons needed
        let comparisons = assert_eq_values(compiler, lhs_info, rhs_info, lhs_type, span);

        // then add all the constraints
        for comparison in comparisons {
            match comparison {
                Comparison::Vars(lhs, rhs) => {
                    compiler.backend.assert_eq_var(&lhs, &rhs, span);
                }
                Comparison::VarConst(var, constant) => {
                    compiler.backend.assert_eq_const(&var, constant, span);
                }
                Comparison::Constants(a, b) => {
                    if a != b {
                        return Err(Error::new(
                            "constraint-generation",
                            ErrorKind::AssertionFailed,
                            span,
                        ));
                    }
                }
            }
        }

        Ok(None)
    }
}

impl Builtin for AssertFn {
    const SIGNATURE: &'static str = "assert(condition: Bool)";

    /// Asserts that a condition is true.
    fn builtin<B: Backend>(
        compiler: &mut CircuitWriter<B>,
        _generics: &GenericParameters,
        vars: &[VarInfo<B::Field, B::Var>],
        span: Span,
    ) -> Result<Option<Var<<B as Backend>::Field, <B as Backend>::Var>>> {
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
}

impl Builtin for LogFn {
    // todo: currently only supports a single field var
    // to support all the types, we can bypass the type check for this log function for now
    const SIGNATURE: &'static str = "log(var: Field)";

    /// Logging
    fn builtin<B: Backend>(
        compiler: &mut CircuitWriter<B>,
        _generics: &GenericParameters,
        vars: &[VarInfo<B::Field, B::Var>],
        span: Span,
    ) -> Result<Option<Var<B::Field, B::Var>>> {
        for var in vars {
            // todo: will need to support string argument in order to customize msg
            compiler.backend.log_var(var, span);
        }

        Ok(None)
    }
}

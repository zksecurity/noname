use std::{collections::HashMap, ops::Neg};

use once_cell::sync::Lazy;

use ark_ff::{One, Zero};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{FnHandle, FnKind},
    lexer::Token,
    parser::{
        types::{FnSig, TyKind},
        ParserCtx,
    },
    type_checker::FnInfo,
    var::{ConstOrCell, Var},
};

pub mod crypto;

/// Takes a list of function signatures (as strings) and their associated function pointer,
/// returns the same list but with the parsed functions (as [FunctionSig]).
pub fn parse_fn_sigs<B: Backend>(fn_sigs: &[(&str, FnHandle<B>)]) -> HashMap<String, FnInfo<B>> {
    let mut functions = HashMap::new();
    let ctx = &mut ParserCtx::default();

    for (sig, fn_ptr) in fn_sigs {
        // filename_id 0 is for builtins
        let mut tokens = Token::parse(0, sig).unwrap();

        let sig = FnSig::parse(ctx, &mut tokens).unwrap();

        functions.insert(
            sig.name.value.clone(),
            FnInfo {
                kind: FnKind::BuiltIn(sig, *fn_ptr),
                span: Span::default(),
            },
        );
    }

    functions
}

//
// Builtins or utils (imported by default)
// TODO: give a name that's useful for the user,
//       not something descriptive internally like "builtins"

pub const QUALIFIED_BUILTINS: &str = "std/builtins";

const ASSERT_FN: &str = "assert(condition: Bool)";
const ASSERT_EQ_FN: &str = "assert_eq(lhs: Field, rhs: Field)";

pub static BUILTIN_FNS_SIGS: Lazy<HashMap<&'static str, FnSig>> = Lazy::new(|| {
    let sigs = [ASSERT_FN, ASSERT_EQ_FN];

    // create a hashmap from the FnSig
    sigs.iter()
        .map(|sig| {
            let ctx = &mut ParserCtx::default();
            let mut tokens = Token::parse(0, sig).unwrap();
            let fn_sig = FnSig::parse(ctx, &mut tokens).unwrap();

            (sig.to_owned(), fn_sig)
        })
        .collect()
});

pub fn get_builtin_fn<B>(name: &str) -> FnInfo<B>
where
    B: Backend,
{
    let ctx = &mut ParserCtx::default();
    let mut tokens = Token::parse(0, name).unwrap();
    let sig = FnSig::parse(ctx, &mut tokens).unwrap();

    let fn_handle = match name {
        ASSERT_FN => assert,
        ASSERT_EQ_FN => assert_eq,
        _ => unreachable!(),
    };

    FnInfo {
        kind: FnKind::BuiltIn(sig, fn_handle),
        span: Span::default(),
    }
}

/// a function iterate through builtin functions
pub fn builtin_fns<B: Backend>() -> Vec<FnInfo<B>> {
    BUILTIN_FNS_SIGS
        .iter()
        .map(|(sig, _)| get_builtin_fn::<B>(sig))
        .collect()
}

pub fn has_builtin_fn(name: &str) -> bool {
    BUILTIN_FNS_SIGS.iter().any(|(_, s)| s.name.value == name)
}

fn assert_eq<B: Backend>(
    circuit: &mut CircuitWriter<B>,
    vars: &[VarInfo<B::Field>],
    span: Span,
) -> Result<Option<Var<B::Field>>> {
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
            circuit.backend.add_generic_gate(
                "constrain var - cst = 0 to check equality",
                vec![Some(*cvar)],
                vec![
                    B::Field::one(),
                    B::Field::zero(),
                    B::Field::zero(),
                    B::Field::zero(),
                    cst.neg(),
                ],
                span,
            );
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // TODO: use permutation to check that
            circuit.backend.add_generic_gate(
                "constrain lhs - rhs = 0 to assert that they are equal",
                vec![Some(*lhs), Some(*rhs)],
                vec![B::Field::one(), B::Field::one().neg()],
                span,
            );
        }
    }

    Ok(None)
}

fn assert<B: Backend>(
    circuit: &mut CircuitWriter<B>,
    vars: &[VarInfo<B::Field>],
    span: Span,
) -> Result<Option<Var<B::Field>>> {
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
            // TODO: use permutation to check that
            let zero = B::Field::zero();
            let one = B::Field::one();
            circuit.backend.add_generic_gate(
                "constrain 1 - X = 0 to assert that X is true",
                vec![None, Some(*cvar)],
                // use the constant to constrain 1 - X = 0
                vec![zero, one.neg(), zero, zero, one],
                span,
            );
        }
    }

    Ok(None)
}

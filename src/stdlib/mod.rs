use std::collections::HashSet;

use ark_ff::{One, Zero};
use kimchi::o1_utils::FieldHelpers;
use once_cell::sync::Lazy;

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    constraints::boolean,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    lexer::Token,
    parser::{
        types::{FnSig, GenericParameters, TyKind},
        ParserCtx,
    },
    type_checker::FnInfo,
    var::{ConstOrCell, Value, Var},
};

pub mod crypto;

//
// Builtins or utils (imported by default)
// TODO: give a name that's useful for the user,
//       not something descriptive internally like "builtins"

pub const QUALIFIED_BUILTINS: &str = "std/builtins";

const ASSERT_FN: &str = "assert(condition: Bool)";
const ASSERT_EQ_FN: &str = "assert_eq(lhs: Field, rhs: Field)";
const TO_BITS_FN: &str = "to_bits(const LEN: Field, val: Field) -> [Bool; LEN]";
const FROM_BITS_FN: &str = "from_bits(bits: [Bool; LEN]) -> Field";

// todo: each addition of builtins require changing this and `get_builtin_fn`,
// - can we encapsulate them in a single place?
/// List of builtin function signatures.
pub const BUILTIN_SIGS: &[&str] = &[ASSERT_FN, ASSERT_EQ_FN, TO_BITS_FN, FROM_BITS_FN];

// Unique set of builtin function names, derived from function signatures.
pub static BUILTIN_FN_NAMES: Lazy<HashSet<String>> = Lazy::new(|| {
    BUILTIN_SIGS
        .iter()
        .map(|s| {
            let ctx = &mut ParserCtx::default();
            let mut tokens = Token::parse(0, s).unwrap();
            let sig = FnSig::parse(ctx, &mut tokens).unwrap();
            sig.name.value
        })
        .collect()
});

pub fn get_builtin_fn<B: Backend>(name: &str) -> Option<FnInfo<B>> {
    let ctx = &mut ParserCtx::default();
    let mut tokens = Token::parse(0, name).unwrap();
    let sig = FnSig::parse(ctx, &mut tokens).unwrap();

    let fn_handle = match name {
        ASSERT_FN => assert,
        ASSERT_EQ_FN => assert_eq,
        TO_BITS_FN => to_bits,
        FROM_BITS_FN => from_bits,
        _ => return None,
    };

    Some(FnInfo {
        kind: FnKind::BuiltIn(sig, fn_handle),
        span: Span::default(),
    })
}

/// a function returns builtin functions
pub fn builtin_fns<B: Backend>() -> Vec<FnInfo<B>> {
    BUILTIN_SIGS
        .iter()
        .map(|sig| get_builtin_fn(sig).unwrap())
        .collect()
}

/// Asserts that two vars are equal.
fn assert_eq<B: Backend>(
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
fn assert<B: Backend>(
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

fn to_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // should be two input vars
    assert_eq!(vars.len(), 2);

    // but the better practice would be to retrieve the value from the generics
    let bitlen = generics.get("LEN") as usize;

    // num should be greater than 0
    assert!(bitlen > 0);

    let modulus_bits: usize = B::Field::modulus_biguint()
        .bits()
        .try_into()
        .expect("modulus is too large");

    assert!(bitlen <= (modulus_bits - 1));

    // alternatively, it can be retrieved from the first var, but it is not recommended
    // let num_var = &vars[0];

    // second var is the value to convert
    let var_info = &vars[1];
    let var = &var_info.var;
    assert_eq!(var.len(), 1);

    let val = match &var[0] {
        ConstOrCell::Cell(cvar) => cvar.clone(),
        ConstOrCell::Const(cst) => {
            // extract the first bitlen bits
            let bits = cst
                .to_bits()
                .iter()
                .take(bitlen)
                .copied()
                // convert to ConstOrVar
                .map(|b| ConstOrCell::Const(B::Field::from(b)))
                .collect::<Vec<_>>();

            return Ok(Some(Var::new(bits, span)));
        }
    };

    // convert value to bits
    let mut bits = Vec::with_capacity(bitlen);
    let mut e2 = B::Field::one();
    let mut lc: Option<B::Var> = None;

    for i in 0..bitlen {
        let bit = compiler
            .backend
            .new_internal_var(Value::NthBit(val.clone(), i), span);

        // constrain it to be either 0 or 1
        // bits[i] * (bits[i] - 1 ) === 0;
        boolean::check(compiler, &ConstOrCell::Cell(bit.clone()), span);

        // lc += bits[i] * e2;
        let weighted_bit = compiler.backend.mul_const(&bit, &e2, span);
        lc = if i == 0 {
            Some(weighted_bit)
        } else {
            Some(compiler.backend.add(&lc.unwrap(), &weighted_bit, span))
        };

        bits.push(bit.clone());
        e2 = e2 + e2;
    }

    compiler.backend.assert_eq_var(&val, &lc.unwrap(), span);

    let bits_cvars = bits.into_iter().map(ConstOrCell::Cell).collect();
    Ok(Some(Var::new(bits_cvars, span)))
}

fn from_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // only one input var
    assert_eq!(vars.len(), 1);

    let var_info = &vars[0];
    let bitlen = generics.get("LEN") as usize;

    let modulus_bits: usize = B::Field::modulus_biguint()
        .bits()
        .try_into()
        .expect("modulus is too large");

    assert!(bitlen <= (modulus_bits - 1));

    let bits_vars: Vec<_> = var_info
        .var
        .cvars
        .iter()
        .map(|c| match c {
            ConstOrCell::Cell(c) => c.clone(),
            ConstOrCell::Const(cst) => {
                // use a cell var to represent the const for now
                // later we will refactor the backend handle ConstOrCell arguments, so we don't have deal with this everywhere
                compiler
                    .backend
                    .add_constant(Some("converted constant"), *cst, span)
            }
        })
        .collect();

    // this might not be necessary since it should be checked in the type checker
    assert_eq!(bitlen, bits_vars.len());

    let mut e2 = B::Field::one();
    let mut lc: Option<B::Var> = None;

    // accumulate the contribution of each bit
    for bit in bits_vars {
        let weighted_bit = compiler.backend.mul_const(&bit, &e2, span);

        lc = match lc {
            None => Some(weighted_bit),
            Some(v) => Some(compiler.backend.add(&v, &weighted_bit, span)),
        };

        e2 = e2 + e2;
    }

    let cvar = ConstOrCell::Cell(lc.unwrap());

    Ok(Some(Var::new_cvar(cvar, span)))
}

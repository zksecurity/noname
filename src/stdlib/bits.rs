use std::vec;

use ark_ff::{BigInteger, One};
use kimchi::{o1_utils::FieldHelpers, turshi::helper::CairoFieldHelpers};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    constraints::boolean,
    error::Result,
    parser::types::{GenericParameters, TyKind},
    var::{ConstOrCell, Value, Var},
};

use super::{FnInfoType, Module};

const TO_BITS_FN: &str = "to_bits(const LEN: Field, val: Field) -> [Bool; LEN]";
const FROM_BITS_FN: &str = "from_bits(bits: [Bool; LEN]) -> Field";

const NTH_BIT_FN: &str = "nth_bit(val: Field, const nth: Field) -> Bool";
const LEFT_SHIFT_FN: &str = "left_shift(val: Field, const shift: Field) -> Field";

const BIT_LEN_FN: &str = "bit_len(const value: Field) -> Field";

pub struct BitsLib {}

impl Module for BitsLib {
    const MODULE: &'static str = "bits";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>)> {
        vec![
            (TO_BITS_FN, to_bits),
            (FROM_BITS_FN, from_bits),
            (NTH_BIT_FN, nth_bit),
            (LEFT_SHIFT_FN, left_shift),
            (BIT_LEN_FN, bit_len),
        ]
    }
}

fn nth_bit<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // should be two input vars
    assert_eq!(vars.len(), 2);

    // these should be type checked already, unless it is called by other low level functions
    // eg. builtins
    let var_info = &vars[0];
    let val = &var_info.var;
    assert_eq!(val.len(), 1);

    let var_info = &vars[1];
    let nth = &var_info.var;
    assert_eq!(nth.len(), 1);

    let nth: usize = match &nth[0] {
        ConstOrCell::Cell(_) => unreachable!("nth should be a constant"),
        ConstOrCell::Const(cst) => cst.to_u64() as usize,
    };

    let val = match &val[0] {
        ConstOrCell::Cell(cvar) => cvar.clone(),
        ConstOrCell::Const(cst) => {
            // directly return the nth bit without adding symbolic value as it doesn't depend on a cell var
            let bit = cst.to_bits();
            return Ok(Some(Var::new_cvar(
                ConstOrCell::Const(B::Field::from(bit[nth])),
                span,
            )));
        }
    };

    // create a cell var for the symbolic value representing the nth bit.
    // it seems we will always have to create cell vars to allocate the symbolic values that involve non-deterministic calculations.
    // it is non-deterministic because it involves non-deterministic arithmetic on a cell var.
    let bit = compiler
        .backend
        .new_internal_var(Value::NthBit(val.clone(), nth), span);

    Ok(Some(Var::new(vec![ConstOrCell::Cell(bit)], span)))
}

fn left_shift<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // should be two input vars
    assert_eq!(vars.len(), 2);

    let var_info = &vars[0];
    let val = &var_info.var;
    assert_eq!(val.len(), 1);

    let var_info = &vars[1];
    let shift = &var_info.var;
    assert_eq!(shift.len(), 1);

    let shift: usize = match &shift[0] {
        ConstOrCell::Cell(_) => unreachable!("shift should be a constant"),
        ConstOrCell::Const(cst) => cst.to_u64() as usize,
    };

    let val = match &val[0] {
        ConstOrCell::Cell(cvar) => cvar.clone(),
        ConstOrCell::Const(cst) => {
            // directly return the nth bit without adding symbolic value as it doesn't depend on a cell var
            let shifted = cst.to_biguint() << shift;
            return Ok(Some(Var::new_cvar(
                ConstOrCell::Const(B::Field::from(shifted)),
                span,
            )));
        }
    };

    // create a cell var for the symbolic value that depends on another cell var.
    let shifted = compiler
        .backend
        .new_internal_var(Value::LeftShift(val.clone(), shift), span);

    Ok(Some(Var::new(vec![ConstOrCell::Cell(shifted)], span)))
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

/// Unconstrained log ceil.
fn bit_len<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // we get a single var
    assert_eq!(vars.len(), 1);

    // of type field
    let var_info = &vars[0];
    if !matches!(var_info.typ, Some(TyKind::Field { .. })) {
        panic!(
            "the var of log_ceil must be of type Field. It was of type {:?}",
            var_info.typ
        );
    }

    // of only one field element
    let var = &var_info.var;
    assert_eq!(var.len(), 1);
    let val = &var[0];

    match val {
        ConstOrCell::Const(cst) => {
            let res = cst.to_biguint().bits();
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

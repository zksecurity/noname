use ark_ff::PrimeField;
use std::vec;

use kimchi::{o1_utils::FieldHelpers, turshi::helper::CairoFieldHelpers};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::types::GenericParameters,
    var::{ConstOrCell, Value, Var},
};

use super::{FnInfoType, Module};

const NTH_BIT_FN: &str = "nth_bit(val: Field, const nth: Field) -> Field";
const CHECK_FIELD_SIZE_FN: &str = "check_field_size(cmp: Field)";

pub struct BitsLib {}

impl Module for BitsLib {
    const MODULE: &'static str = "bits";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![
            (NTH_BIT_FN, nth_bit, false),
            (CHECK_FIELD_SIZE_FN, check_field_size, false),
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

    let bit = compiler
        .backend
        .new_internal_var(Value::NthBit(val.clone(), nth), span);

    Ok(Some(Var::new(vec![ConstOrCell::Cell(bit)], span)))
}

// Ensure that the field size is not exceeded
fn check_field_size<B: Backend>(
    _compiler: &mut CircuitWriter<B>,
    _generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    let var = &vars[0].var[0];
    let bit_len = B::Field::size_in_bits() as u64;

    match var {
        ConstOrCell::Const(cst) => {
            let to_cmp = cst.to_u64();
            if to_cmp >= bit_len {
                return Err(Error::new(
                    "constraint-generation",
                    ErrorKind::AssertionFailed,
                    span,
                ));
            }
            Ok(None)
        }
        ConstOrCell::Cell(_) => Err(Error::new(
            "constraint-generation",
            ErrorKind::ExpectedConstant,
            span,
        )),
    }
}

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

use super::{builtins::Builtin, FnInfoType, Module};

pub struct BitsLib {}

impl Module for BitsLib {
    const MODULE: &'static str = "bits";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![
            (NthBitFn::SIGNATURE, NthBitFn::builtin, false),
            (
                CheckFieldSizeFn::SIGNATURE,
                CheckFieldSizeFn::builtin,
                false,
            ),
        ]
    }
}

struct NthBitFn {}
struct CheckFieldSizeFn {}

impl Builtin for NthBitFn {
    const SIGNATURE: &'static str = "nth_bit(val: Field, const nth: Field) -> Field";

    fn builtin<B: Backend>(
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
}

impl Builtin for CheckFieldSizeFn {
    const SIGNATURE: &'static str = "check_field_size(cmp: Field)";

    fn builtin<B: Backend>(
        _compiler: &mut CircuitWriter<B>,
        _generics: &GenericParameters,
        vars: &[VarInfo<B::Field, B::Var>],
        span: Span,
    ) -> Result<Option<Var<B::Field, B::Var>>> {
        let var = &vars[0].var[0];
        let bit_len = B::Field::MODULUS_BIT_SIZE as u64;

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
}

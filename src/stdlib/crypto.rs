use super::{builtins::Builtin, FnInfoType, Module};
use crate::backends::Backend;

pub struct CryptoLib {}

impl Module for CryptoLib {
    const MODULE: &'static str = "crypto";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![(PoseidonFn::SIGNATURE, PoseidonFn::builtin, false)]
    }
}

struct PoseidonFn {}

impl Builtin for PoseidonFn {
    const SIGNATURE: &'static str = "poseidon(input: [Field; 2]) -> [Field; 3]";

    fn builtin<B: Backend>(
        compiler: &mut crate::circuit_writer::CircuitWriter<B>,
        generics: &crate::parser::types::GenericParameters,
        vars: &[crate::circuit_writer::VarInfo<B::Field, B::Var>],
        span: crate::constants::Span,
    ) -> crate::error::Result<Option<crate::var::Var<B::Field, B::Var>>> {
        B::poseidon()(compiler, generics, vars, span)
    }
}

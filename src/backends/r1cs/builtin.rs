use crate::{
    backends::BackendField,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    var::Var,
};

use super::{LinearCombination, R1CS};

// todo: impl this
pub fn poseidon<F>(
    compiler: &mut CircuitWriter<R1CS<F>>,
    vars: &[VarInfo<F, LinearCombination<F>>],
    span: Span,
) -> Result<Option<Var<F, LinearCombination<F>>>>
where
    F: BackendField,
{
    // dummy for now
    unimplemented!()
}

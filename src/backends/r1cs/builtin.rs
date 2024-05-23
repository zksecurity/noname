use crate::{
    backends::BackendField,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    var::Var,
};

use super::{R1csCellVar, R1CS};

// todo: impl this
pub fn poseidon<F>(
    compiler: &mut CircuitWriter<R1CS<F>>,
    vars: &[VarInfo<F, R1csCellVar<F>>],
    span: Span,
) -> Result<Option<Var<F, R1csCellVar<F>>>> 
where 
    F: BackendField,
{
    // dummy for now
    unimplemented!()
}

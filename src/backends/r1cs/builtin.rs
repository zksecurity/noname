use ark_bls12_381::Fr;

use crate::{
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    var::Var,
};

use super::R1csBls12_381;

// todo: impl this
pub fn poseidon(
    compiler: &mut CircuitWriter<R1csBls12_381>,
    vars: &[VarInfo<Fr>],
    span: Span,
) -> Result<Option<Var<Fr>>> {
    // dummy for now
    unimplemented!()
}

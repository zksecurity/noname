use std::collections::HashMap;
use std::sync::Arc;

use kimchi::circuits::polynomials::poseidon::{POS_ROWS_PER_HASH, ROUNDS_PER_ROW};
use kimchi::mina_poseidon::constants::{PlonkSpongeConstantsKimchi, SpongeConstants};
use kimchi::mina_poseidon::pasta::fp_kimchi::params;
use kimchi::mina_poseidon::permutation::full_round;

use crate::backends::{self, Backend};
use crate::error::Error;
use crate::imports::FnKind;
use crate::lexer::Token;
use crate::parser::types::FnSig;
use crate::parser::ParserCtx;
use crate::type_checker::FnInfo;
use crate::{
    circuit_writer::{CircuitWriter, GateKind, VarInfo},
    constants::{self, Field, Span},
    error::{ErrorKind, Result},
    parser::types::TyKind,
    var::{ConstOrCell, Value, Var},
};
use ark_ff::Zero;

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub const CRYPTO_SIGS: &[&str] = &[POSEIDON_FN];

pub fn get_crypto_fn<B: Backend>(name: &str) -> Option<FnInfo<B>> {
    let ctx = &mut ParserCtx::default();
    let mut tokens = Token::parse(0, name).unwrap();
    let sig = FnSig::parse(ctx, &mut tokens).unwrap();

    let fn_handle = match name {
        POSEIDON_FN => B::poseidon(),
        _ => return None,
    };

    Some(FnInfo {
        kind: FnKind::BuiltIn(sig, fn_handle),
        span: Span::default(),
    })
}

/// a function returns crypto functions
pub fn crypto_fns<B: Backend>() -> Vec<FnInfo<B>> {
    CRYPTO_SIGS
        .iter()
        .map(|sig| get_crypto_fn(sig).unwrap())
        .collect()
}

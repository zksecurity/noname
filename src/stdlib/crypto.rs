use crate::backends::Backend;
use crate::constants::Span;
use crate::imports::FnKind;
use crate::lexer::Token;
use crate::parser::types::FnSig;
use crate::parser::ParserCtx;
use crate::type_checker::FnInfo;

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub const CRYPTO_SIGS: &[&str] = &[POSEIDON_FN];

#[must_use]
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
#[must_use]
pub fn crypto_fns<B: Backend>() -> Vec<FnInfo<B>> {
    CRYPTO_SIGS
        .iter()
        .map(|sig| get_crypto_fn(sig).unwrap())
        .collect()
}

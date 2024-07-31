use crate::backends::Backend;
use crate::constants::Span;
use crate::imports::FnKind;
use crate::lexer::Token;
use crate::parser::types::FnSig;
use crate::parser::ParserCtx;
use crate::type_checker::FnInfo;

use super::{FnInfoType, Module};

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub struct CryptoLib {}

impl Module for CryptoLib {
    const MODULE: &'static str = "crypto";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>)> {
        vec![(POSEIDON_FN, B::poseidon())]
    }
}

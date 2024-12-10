use super::{FnInfoType, Module};
use crate::backends::Backend;

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub struct CryptoLib {}

impl Module for CryptoLib {
    const MODULE: &'static str = "crypto";

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>, bool)> {
        vec![(POSEIDON_FN, B::poseidon(), false)]
    }
}

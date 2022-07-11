//! noname project

pub mod asm;
pub mod ast;
pub mod constants;
pub mod error;
pub mod field;
pub mod lexer;
pub mod parser;
pub mod prover;
pub mod stdlib;
pub mod tokens;
pub mod type_checker;
pub mod witness;

#[cfg(test)]
pub mod tests;

//
// Helpers
//

pub mod helpers {
    use kimchi::oracle::{constants::PlonkSpongeConstantsKimchi, poseidon::Sponge};

    use crate::field::Field;

    pub fn poseidon(input: [Field; 2]) -> Field {
        let mut sponge: kimchi::oracle::poseidon::ArithmeticSponge<
            Field,
            PlonkSpongeConstantsKimchi,
        > = kimchi::oracle::poseidon::ArithmeticSponge::new(
            kimchi::oracle::pasta::fp_kimchi::params(),
        );
        sponge.absorb(&input);
        sponge.squeeze()
    }
}

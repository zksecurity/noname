//! This is a high-level language to write circuits that you can prove in kimchi.
//! Refer to the [book](https://mimoo.github.io/noname/) for more information.
//!

pub mod asm;
pub mod boolean;
pub mod circuit_writer;
pub mod compiler;
pub mod constants;
pub mod error;
pub mod field;
pub mod imports;
pub mod inputs;
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

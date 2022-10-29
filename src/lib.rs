//! This is a high-level language to write circuits that you can prove in kimchi.
//! Refer to the [book](https://mimoo.github.io/noname/) for more information.
//!

pub mod asm;
pub mod boolean;
pub mod circuit_writer;
pub mod cli;
pub mod compiler;
pub mod constants;
pub mod error;
pub mod field;
pub mod imports;
pub mod inputs;
pub mod lexer;
pub mod parser;
pub mod prover;
pub mod serialization;
pub mod stdlib;
pub mod syntax;
pub mod tokens;
pub mod type_checker;
pub mod var;
pub mod witness;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod negative_tests;

//
// Helpers
//

pub mod helpers {
    use kimchi::oracle::{constants::PlonkSpongeConstantsKimchi, poseidon::Sponge};

    use crate::constants::Field;

    /// A trait to display [Field] in pretty ways.
    pub trait PrettyField: ark_ff::PrimeField {
        /// Print a field in a negative form if it's past the half point.
        fn pretty(&self) -> String {
            let bigint: num_bigint::BigUint = (*self).into();
            let inv: num_bigint::BigUint = self.neg().into(); // gettho way of splitting the field into positive and negative elements
            if inv < bigint {
                format!("-{}", inv)
            } else {
                bigint.to_string()
            }
        }
    }

    impl PrettyField for Field {}

    pub fn poseidon(input: [Field; 2]) -> Field {
        let mut sponge: kimchi::oracle::poseidon::ArithmeticSponge<
            Field,
            PlonkSpongeConstantsKimchi,
        > = kimchi::oracle::poseidon::ArithmeticSponge::new(
            kimchi::oracle::pasta::fp_kimchi::static_params(),
        );
        sponge.absorb(&input);
        sponge.squeeze()
    }
}

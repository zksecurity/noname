//! This is a high-level language to write circuits that you can prove in kimchi.
//! Refer to the [book](https://mimoo.github.io/noname/) for more information.
//!

pub mod backends;
pub mod circuit_writer;
pub mod cli;
pub mod compiler;
pub mod constants;
pub mod constraints;
pub mod error;
pub mod imports;
pub mod inputs;
pub mod lexer;
pub mod name_resolution;
pub mod parser;
pub mod serialization;
pub mod stdlib;
pub mod syntax;
pub mod type_checker;
pub mod utils;
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
    use kimchi::mina_poseidon::{
        constants::PlonkSpongeConstantsKimchi,
        pasta::fp_kimchi,
        poseidon::{ArithmeticSponge, Sponge},
    };

    use crate::backends::{
        kimchi::VestaField,
        r1cs::{R1csBls12381Field, R1csBn254Field},
    };

    /// A trait to display [Field] in pretty ways.
    pub trait PrettyField: ark_ff::PrimeField {
        /// Print a field in a negative form if it's past the half point.
        fn pretty(&self) -> String {
            let bigint: num_bigint::BigUint = (*self).into();
            let inv: num_bigint::BigUint = self.neg().into(); // gettho way of splitting the field into positive and negative elements
            if inv < bigint {
                format!("-{inv}")
            } else {
                bigint.to_string()
            }
        }
    }

    impl PrettyField for VestaField {}
    impl PrettyField for R1csBls12381Field {}
    impl PrettyField for R1csBn254Field {}

    #[must_use]
    pub fn poseidon(input: [VestaField; 2]) -> VestaField {
        let mut sponge: ArithmeticSponge<VestaField, PlonkSpongeConstantsKimchi> =
            ArithmeticSponge::new(fp_kimchi::static_params());
        sponge.absorb(&input);
        sponge.squeeze()
    }
}

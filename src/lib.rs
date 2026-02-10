//! This is a high-level language to write circuits that you can prove in kimchi.
//! Refer to the [book](https://zksecurity.github.io/noname/) for more information.
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
pub mod mast;
pub mod name_resolution;
pub mod parser;
pub mod serialization;
pub mod server;
pub mod stdlib;
pub mod syntax;
pub mod type_checker;
pub mod utils;
pub mod var;
pub mod witness;

#[cfg(test)]
pub mod tests;

#[cfg(all(test, feature = "r1cs"))]
pub mod negative_tests;

//
// Helpers
//

pub mod helpers {
    #[cfg(feature = "kimchi")]
    use kimchi::mina_poseidon::{
        constants::PlonkSpongeConstantsKimchi,
        pasta::fp_kimchi,
        poseidon::{ArithmeticSponge, Sponge},
    };

    #[cfg(feature = "kimchi")]
    use crate::backends::kimchi::VestaField;
    #[cfg(feature = "r1cs")]
    use crate::backends::r1cs::{R1csBls12381Field, R1csBn254Field};

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

    #[cfg(feature = "kimchi")]
    impl PrettyField for VestaField {}
    #[cfg(feature = "r1cs")]
    impl PrettyField for R1csBls12381Field {}
    #[cfg(feature = "r1cs")]
    impl PrettyField for R1csBn254Field {}

    #[cfg(feature = "kimchi")]
    pub fn poseidon(input: [VestaField; 2]) -> VestaField {
        let mut sponge: ArithmeticSponge<VestaField, PlonkSpongeConstantsKimchi> =
            ArithmeticSponge::new(fp_kimchi::static_params());
        sponge.absorb(&input);
        sponge.squeeze()
    }
}

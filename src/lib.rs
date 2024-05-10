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
    use std::fmt::Write;

    use crate::backends::kimchi::VestaField;

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

    impl PrettyField for VestaField {}
    impl PrettyField for ark_bls12_381::Fr {}

    pub fn poseidon(input: [VestaField; 2]) -> VestaField {
        let mut sponge: ArithmeticSponge<VestaField, PlonkSpongeConstantsKimchi> =
            ArithmeticSponge::new(fp_kimchi::static_params());
        sponge.absorb(&input);
        sponge.squeeze()
    }

    pub fn noname_version() -> String {
        format!("@ noname.{}\n\n", env!("CARGO_PKG_VERSION"))
    }

    pub fn display_source(
        res: &mut String,
        sources: &crate::compiler::Sources,
        debug_infos: &[crate::circuit_writer::DebugInfo],
    ) {
        for crate::circuit_writer::DebugInfo { span, note: _ } in debug_infos {
            // find filename and source
            let (file, source) = sources.get(&span.filename_id).expect("source not found");
    
            // top corner
            res.push('╭');
            res.push_str(&"─".repeat(80));
            res.push('\n');
    
            // display filename
            writeln!(res, "│ FILE: {}", file).unwrap();
            writeln!(res, "│{s}", s = "─".repeat(80)).unwrap();
    
            // source
            res.push_str("│ ");
            let (line_number, start, line) = find_exact_line(source, *span);
            let header = format!("{line_number}: ");
            writeln!(res, "{header}{line}").unwrap();
    
            // caption
            res.push('│');
            res.push_str(&" ".repeat(header.len() + 1 + span.start - start));
            res.push_str(&"^".repeat(span.len));
            res.push('\n');
        }
    
        // bottom corner
        res.push('╰');
        res.push_str(&"─".repeat(80));
        res.push('\n');
    }

    fn find_exact_line(source: &str, span: crate::constants::Span) -> (usize, usize, &str) {
        let ss = source.as_bytes();
        let mut start = span.start;
        let mut end = span.end();
        while start > 0 && (ss[start - 1] as char) != '\n' {
            start -= 1;
        }
        while end < source.len() && (ss[end] as char) != '\n' {
            end += 1;
        }
    
        let line = &source[start..end];
    
        let line_number = source[..start].matches('\n').count() + 1;
    
        (line_number, start, line)
    }

    pub fn title(res: &mut String, s: &str) {
        writeln!(res, "╭{s}╮", s = "─".repeat(s.len())).unwrap();
        writeln!(res, "│{s}│", s = s).unwrap();
        writeln!(res, "╰{s}╯", s = "─".repeat(s.len())).unwrap();
        writeln!(res).unwrap();
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn find_lines() {
            const SRC: &str = "abcd
efgh
ijkl
mnop
qrst
uvwx
yz
";
            assert_eq!(
                find_exact_line(SRC, crate::constants::Span::new(0, 5, 6)),
                (2, 5, "efgh\nijkl")
            );
        }
    }
}

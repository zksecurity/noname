//! ASM-like language:
//!
//! ```
//! @ noname.0.1.0
//! # gates
//!
//! DoubleGeneric<1>
//! DoubleGeneric<1,1,-1>
//! DoubleGeneric<1,0,0,0,-2>
//! DoubleGeneric<1,-1>
//!
//! # wiring
//!
//! (2,0) -> (3,1)
//! (1,2) -> (3,0)
//! (0,0) -> (1,1)
//! ```

use std::{collections::HashMap, ops::Neg};

use itertools::Itertools;
use num_bigint::BigUint;

use crate::{
    ast::{CellVar, Gate, Wiring},
    constants::Span,
    field::{Field, PrettyField as _},
};

pub fn generate_asm(
    source: &str,
    gates: &[Gate],
    wiring: &HashMap<CellVar, Wiring>,
    debug: bool,
) -> String {
    let mut res = "".to_string();

    // version
    res.push_str("@ noname.0.1.0\n\n");

    if debug {
        res.push_str("# gates\n\n");
    }

    // gates
    for Gate { typ, coeffs, span } in gates {
        // source
        if debug {
            res.push_str(&"-".repeat(80));
            res.push_str("\n");
            let (line_number, start, line) = find_exact_line(source, *span);
            let header = format!("{line_number}: ");
            res.push_str(&format!("{header}{line}\n"));
            res.push_str(&" ".repeat(header.len() + span.0 - start));
            res.push_str(&"^".repeat(span.1));
            res.push_str("\n");
        }

        // gate coeffs
        let (vars, coeffs) = parse_coeffs(coeffs);
        res.push_str(&vars);

        // gate
        res.push_str(&format!("{typ:?}"));
        res.push_str("<");
        res.push_str(&coeffs.join(","));
        res.push_str(">\n");
    }

    // wiring
    if debug {
        res.push_str("\n# wiring\n\n");
    }

    let mut cycles: Vec<_> = wiring
        .values()
        .map(|w| match w {
            Wiring::NotWired(_) => None,
            Wiring::Wired(cells) => Some(cells),
        })
        .filter(Option::is_some)
        .flatten()
        .collect();

    // we must have a deterministic sort for the cycles,
    // otherwise the same circuit might have different representations
    cycles.sort();

    for cells in cycles {
        let s = cells.iter().map(|cell| format!("{cell}")).join(" -> ");
        res.push_str(&format!("{s}\n"));
    }

    res
}

fn parse_coeffs(coeffs: &[Field]) -> (String, Vec<String>) {
    let mut vars = String::new();
    let coeffs = coeffs
        .iter()
        .map(|x| {
            let s = x.pretty();
            if s.len() < 5 {
                s
            } else {
                let var = format!("c{}", vars.len());
                vars.push_str(&format!("{var}={s}\n"));
                var
            }
        })
        .collect();
    (vars, coeffs)
}

fn find_exact_line(source: &str, span: Span) -> (usize, usize, &str) {
    let ss = source.as_bytes();
    let mut start = span.0;
    let mut end = span.0 + span.1;
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
        assert_eq!(find_exact_line(&SRC, (5, 6)), (2, 5, "efgh\nijkl"));
    }
}

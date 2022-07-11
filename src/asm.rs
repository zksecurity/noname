//! ASM-like language:
//!
//! ```
//! @ noname.0.1.0
//!
//! # vars
//!
//! c0 = -9352361074401710304385665936723449560966553519198046749109814779611130548623
//! # gates
//!
//! DoubleGeneric<c0>
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
//!

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use itertools::Itertools;

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

    // vars
    let mut vars = OrderedHashSet::new();

    for Gate { coeffs, .. } in gates {
        extract_vars_from_coeffs(&mut vars, coeffs);
    }

    for (idx, var) in vars.iter().enumerate() {
        res.push_str(&format!("c{idx} = {}\n", var.pretty()));
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
        let coeffs = parse_coeffs(&vars, coeffs);

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

fn extract_vars_from_coeffs(vars: &mut OrderedHashSet<Field>, coeffs: &[Field]) {
    for coeff in coeffs {
        let s = coeff.pretty();
        if s.len() >= 5 {
            vars.insert(*coeff);
        }
    }
}

fn parse_coeffs(vars: &OrderedHashSet<Field>, coeffs: &[Field]) -> Vec<String> {
    coeffs
        .iter()
        .map(|x| {
            let s = x.pretty();
            if s.len() < 5 {
                s
            } else {
                let var_idx = vars.pos(x);
                format!("c{var_idx}")
            }
        })
        .collect()
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

/// Very dumb way to write an ordered hash set.
pub struct OrderedHashSet<T> {
    inner: HashSet<T>,
    map: HashMap<T, usize>,
    ordered: Vec<T>,
}

impl<T> OrderedHashSet<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
            map: HashMap::new(),
            ordered: Vec::new(),
        }
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.inner.insert(value.clone()) {
            self.map.insert(value.clone(), self.ordered.len());
            self.ordered.push(value);
            true
        } else {
            false
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.ordered.iter()
    }

    pub fn pos(&self, value: &T) -> usize {
        self.map[value]
    }
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

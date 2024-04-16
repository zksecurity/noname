//! ASM-like language:
//!
//! ```ignore
//! @ noname.0.7.0
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
use std::fmt::Write;
use std::hash::Hash;

use crate::circuit_writer::{DebugInfo, Gate};
use crate::constants::Span;
use crate::helpers::PrettyField;

use super::{VestaField, KimchiVesta};

pub fn display_source(
    res: &mut String,
    sources: &crate::compiler::Sources,
    debug_infos: &[DebugInfo],
) {
    for DebugInfo { span, note: _ } in debug_infos {
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

pub fn extract_vars_from_coeffs(
    vars: &mut OrderedHashSet<VestaField>,
    coeffs: &[VestaField],
) {
    for coeff in coeffs {
        let s = coeff.pretty();
        if s.len() >= 5 {
            vars.insert(*coeff);
        }
    }
}

pub fn parse_coeffs(vars: &OrderedHashSet<VestaField>, coeffs: &[VestaField]) -> Vec<String> {
    let mut coeffs: Vec<_> = coeffs
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
        // trim trailing zeros
        .rev()
        .skip_while(|x| x == "0")
        .collect();

    coeffs.reverse();

    coeffs
}

pub fn find_exact_line(source: &str, span: Span) -> (usize, usize, &str) {
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

/// Very dumb way to write an ordered hash set.
#[derive(Default)]
pub struct OrderedHashSet<T> {
    inner: HashSet<T>,
    map: HashMap<T, usize>,
    ordered: Vec<T>,
}

impl<T> OrderedHashSet<T>
where
    T: Eq + Hash + Clone,
{
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

    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
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
        assert_eq!(
            find_exact_line(SRC, Span::new(0, 5, 6)),
            (2, 5, "efgh\nijkl")
        );
    }
}
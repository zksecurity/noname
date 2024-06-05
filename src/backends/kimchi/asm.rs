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
use std::hash::Hash;

use crate::helpers::PrettyField;

use super::VestaField;

pub fn extract_vars_from_coeffs(vars: &mut OrderedHashSet<VestaField>, coeffs: &[VestaField]) {
    for coeff in coeffs {
        let s = coeff.pretty();
        if s.len() >= 5 {
            vars.insert(*coeff);
        }
    }
}

#[must_use]
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

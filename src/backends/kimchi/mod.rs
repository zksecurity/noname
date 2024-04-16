use std::collections::HashMap;

use crate::{constants::{Field, Span}, var::{CellVar, Value}};

use super::Backend;
pub mod builtin;

#[derive(Clone, Default)]
pub struct KimchiVesta {
    /// This is used to give a distinct number to each variable during circuit generation.
    pub(crate) next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub(crate) witness_vars: HashMap<usize, Value<Self>>,
}

impl Backend for KimchiVesta {
    type Field = Field;

    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon
    }

    fn witness_vars(&self) -> &HashMap<usize, Value<Self>> {
        &self.witness_vars
    }

    fn new_internal_var(&mut self, val: Value<KimchiVesta>, span: Span) -> CellVar {
        // create new var
        let var = CellVar::new(self.next_variable, span);
        self.next_variable += 1;

        // store it in the circuit_writer
        self.witness_vars.insert(var.index, val);

        var
    }
}
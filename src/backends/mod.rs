use std::collections::HashMap;

use ark_ff::Field;

use crate::{circuit_writer::DebugInfo, constants::Span, helpers::PrettyField, imports::FnHandle, var::{CellVar, Value}};

pub mod kimchi;

// TODO: should it be cloneable? It is now so because FnInfo needs to be cloneable.
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: Field + PrettyField;

    /// This provides a standard way to access to all the internal vars.
    /// Different backends should be accessible in the same way by the variable index.
    fn witness_vars(&self) -> &HashMap<usize, Value<Self>>;

    // TODO: as the builtins grows, we might better change this to a crypto struct that holds all the builtin function pointers.
    /// poseidon crypto builtin function for different backends
    fn poseidon() -> FnHandle<Self>;

    /// Create a new cell variable and record it.
    /// It increments the variable index for look up later.
    fn new_internal_var(&mut self, val: Value<Self>, span: Span) -> CellVar;

    fn debug_info(&self) -> &[DebugInfo];
}
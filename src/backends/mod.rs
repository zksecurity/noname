use std::collections::HashMap;

use ark_ff::Field;

use crate::{
    circuit_writer::{DebugInfo, GateKind},
    constants::Span,
    error::Result,
    helpers::PrettyField,
    imports::FnHandle,
    var::{CellVar, Value, Var},
};

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

    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Self::Field,
        span: Span,
    ) -> CellVar;

    fn debug_info(&self) -> &[DebugInfo];
    /// Add a gate to the circuit. Kimchi specific atm.
    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    /// Add a generic double gate to the circuit. Kimchi specific atm.
    fn add_generic_gate(
        &mut self,
        label: &'static str,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    // TODO: we may need to move the finalized flag from circuit writer to backend, so the backend can freeze itself once finalized.
    /// Finalize the circuit by doing some sanitizing checks.
    fn finalize_circuit(
        &mut self,
        public_output: Option<Var<Self::Field>>,
        returned_cells: Option<Vec<CellVar>>,
        private_input_indices: Vec<usize>,
        main_span: Span,
    ) -> Result<()>;
}

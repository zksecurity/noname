use ark_ff::Field;

use crate::{
    circuit_writer::{writer::AnnotatedCell, DebugInfo, Gate, GateKind},
    constants::Span,
    helpers::PrettyField,
    var::CellVar,
};

pub mod kimchi;
pub mod r1cs;

/// Each backend implements this trait
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: Field + PrettyField;

    // TODO: can we move this back to the CircuitWriter? the implementation seems to be reusable across backends
    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    fn add_constraint(
        &mut self,
        label: &'static str,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    fn gates(&self) -> &[Gate<Self>];

    fn rows_of_vars(&self) -> Vec<Vec<Option<CellVar>>>
    where
        Self: Sized;

    fn wiring_cycles(&self) -> Vec<&Vec<AnnotatedCell>>;

    fn debug_info(&self) -> &[DebugInfo];
}

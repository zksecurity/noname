pub mod fp_kimchi;

use std::{collections::HashMap, fmt::Debug};

use kimchi::circuits::polynomials::generic::{GENERIC_COEFFS, GENERIC_REGISTERS};
use num_traits::Zero;

use crate::{
    circuit_writer::{
        writer::{AnnotatedCell, Cell, PendingGate},
        DebugInfo, Gate, GateKind, Wiring,
    },
    constants::Span,
    var::CellVar,
};

use super::Backend;

/// We use the scalar field of Vesta as our circuit field.
pub type KimchiField = kimchi::mina_curves::pasta::Fp;

/// Number of columns in the execution trace.
pub const NUM_REGISTERS: usize = kimchi::circuits::wires::COLUMNS;

#[derive(Debug, Clone, Default)]
pub struct Kimchi {
    /// The gates created by the circuit generation.
    pub gates: Vec<Gate<Self>>,

    /// The wiring of the circuit.
    /// It is created during circuit generation.
    pub(crate) wiring: HashMap<usize, Wiring>,

    /// If set to false, a single generic gate will be used per double generic gate.
    /// This can be useful for debugging.
    pub(crate) double_generic_gate_optimization: bool,

    /// This is used to implement the double generic gate,
    /// which encodes two generic gates.
    pub(crate) pending_generic_gate: Option<PendingGate<Self>>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub(crate) rows_of_vars: Vec<Vec<Option<CellVar>>>,

    /// A vector of debug information that maps to each row of the created circuit.
    pub(crate) debug_info: Vec<DebugInfo>,
}

impl Backend for Kimchi {
    type Field = KimchiField;

    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<KimchiField>,
        span: Span,
    ) {
        // sanitize
        assert!(coeffs.len() <= NUM_REGISTERS);
        assert!(vars.len() <= NUM_REGISTERS);

        // construct the execution trace with vars, for the witness generation
        self.rows_of_vars.push(vars.clone());

        // get current row
        // important: do that before adding the gate below
        let row = self.gates.len();

        // add gate
        self.gates.push(Gate { typ, coeffs });

        // add debug info related to that gate
        let debug_info = DebugInfo {
            span,
            note: note.to_string(),
        };
        self.debug_info.push(debug_info.clone());

        // wiring (based on vars)
        for (col, var) in vars.iter().enumerate() {
            if let Some(var) = var {
                let curr_cell = Cell { row, col };
                let annotated_cell = AnnotatedCell {
                    cell: curr_cell,
                    debug: debug_info.clone(),
                };

                self.wiring
                    .entry(var.index)
                    .and_modify(|w| match w {
                        Wiring::NotWired(old_cell) => {
                            *w = Wiring::Wired(vec![old_cell.clone(), annotated_cell.clone()])
                        }
                        Wiring::Wired(ref mut cells) => {
                            cells.push(annotated_cell.clone());
                        }
                    })
                    .or_insert(Wiring::NotWired(annotated_cell));
            }
        }
    }

    fn add_constraint(
        &mut self,
        label: &'static str,
        mut vars: Vec<Option<CellVar>>,
        mut coeffs: Vec<KimchiField>,
        span: Span,
    ) {
        // padding
        let coeffs_padding = GENERIC_COEFFS.checked_sub(coeffs.len()).unwrap();
        coeffs.extend(std::iter::repeat(KimchiField::zero()).take(coeffs_padding));

        let vars_padding = GENERIC_REGISTERS.checked_sub(vars.len()).unwrap();
        vars.extend(std::iter::repeat(None).take(vars_padding));

        // if the double gate optimization is not set, just add the gate
        if !self.double_generic_gate_optimization {
            self.add_gate(label, GateKind::DoubleGeneric, vars, coeffs, span);
            return;
        }

        // only add a double generic gate if we have two of them
        if let Some(generic_gate) = self.pending_generic_gate.take() {
            coeffs.extend(generic_gate.coeffs);
            vars.extend(generic_gate.vars);

            // TODO: what to do with the label and span?

            self.add_gate(label, GateKind::DoubleGeneric, vars, coeffs, span);
        } else {
            // otherwise queue it
            self.pending_generic_gate = Some(PendingGate {
                label,
                coeffs,
                vars,
                span,
            });
        }
    }

    fn rows_of_vars(&self) -> Vec<Vec<Option<CellVar>>>
    where
        Self: Sized,
    {
        self.rows_of_vars.clone()
    }

    fn gates(&self) -> &[Gate<Self>] {
        &self.gates
    }

    fn wiring_cycles(&self) -> Vec<&Vec<AnnotatedCell>> {
        self.wiring
            .values()
            .map(|w| match w {
                Wiring::NotWired(_) => None,
                Wiring::Wired(annotated_cells) => Some(annotated_cells),
            })
            .filter(Option::is_some)
            .flatten()
            .collect()
    }

    fn debug_info(&self) -> &[DebugInfo] {
        &self.debug_info
    }
}

use std::collections::HashMap;

use kimchi::circuits::polynomials::generic::{GENERIC_COEFFS, GENERIC_REGISTERS};

use crate::{
    circuit_writer::{
        writer::{AnnotatedCell, Cell, PendingGate},
        DebugInfo, Gate, GateKind, Wiring,
    },
    constants::{Field, Span, NUM_REGISTERS},
    var::{CellVar, Value},
};

use ark_ff::Zero;

use super::Backend;
pub mod builtin;

#[derive(Clone, Default)]
pub struct KimchiVesta {
    /// This is used to give a distinct number to each variable during circuit generation.
    pub(crate) next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub(crate) witness_vars: HashMap<usize, Value<Self>>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub(crate) rows_of_vars: Vec<Vec<Option<CellVar>>>,

    /// The gates created by the circuit generation.
    gates: Vec<Gate>,

    /// The wiring of the circuit.
    /// It is created during circuit generation.
    pub(crate) wiring: HashMap<usize, Wiring>,

    /// If set to false, a single generic gate will be used per double generic gate.
    /// This can be useful for debugging.
    pub(crate) double_generic_gate_optimization: bool,

    /// This is used to implement the double generic gate,
    /// which encodes two generic gates.
    pub(crate) pending_generic_gate: Option<PendingGate>,

    /// A vector of debug information that maps to each row of the created circuit.
    pub(crate) debug_info: Vec<DebugInfo>,
}

impl Backend for KimchiVesta {
    type Field = Field;

    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon
    }

    fn debug_info(&self) -> &[DebugInfo] {
        &self.debug_info
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

    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Field>,
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

    fn add_generic_gate(
        &mut self,
        label: &'static str,
        mut vars: Vec<Option<CellVar>>,
        mut coeffs: Vec<Field>,
        span: Span,
    ) {
        // padding
        let coeffs_padding = GENERIC_COEFFS.checked_sub(coeffs.len()).unwrap();
        coeffs.extend(std::iter::repeat(Field::zero()).take(coeffs_padding));

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
}

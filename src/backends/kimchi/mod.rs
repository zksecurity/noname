pub mod asm;
pub mod builtin;
pub mod prover;

use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    ops::Neg as _,
};

use itertools::{izip, Itertools};
use kimchi::circuits::polynomials::generic::{GENERIC_COEFFS, GENERIC_REGISTERS};
use serde::{Deserialize, Serialize};

use crate::{
    backends::kimchi::asm::parse_coeffs,
    circuit_writer::{
        writer::{AnnotatedCell, Cell, PendingGate},
        DebugInfo, Gate, GateKind, Wiring,
    },
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    var::{Value, Var},
    witness::WitnessEnv,
};

use ark_ff::{One, Zero};

use self::asm::{extract_vars_from_coeffs, OrderedHashSet};

/// We use the scalar field of Vesta as our circuit field.
pub type VestaField = kimchi::mina_curves::pasta::Fp;

/// Number of columns in the execution trace.
pub const NUM_REGISTERS: usize = kimchi::circuits::wires::COLUMNS;

use super::{Backend, BackendField, BackendVar};

impl BackendField for VestaField {}

#[derive(Debug)]
pub struct Witness(Vec<[VestaField; NUM_REGISTERS]>);

// TODO: refine this struct as full_public_inputs and public_outputs overlap with all_witness
pub struct GeneratedWitness {
    /// contains all the witness values
    pub all_witness: Witness,
    /// contains the public inputs, which are also part of the `all_witness`
    pub full_public_inputs: Vec<VestaField>,
    /// contains the public outputs, which are also part of the `all_witness`
    pub public_outputs: Vec<VestaField>,
}

#[derive(Clone)]
pub struct KimchiVesta {
    /// This is used to give a distinct number to each variable during circuit generation.
    pub(crate) next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub(crate) vars_to_value: HashMap<usize, Value<Self>>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub(crate) witness_table: Vec<Vec<Option<KimchiCellVar>>>,

    /// We cache the association between a constant and its _constrained_ variable,
    /// this is to avoid creating a new constraint every time we need to hardcode the same constant.
    pub(crate) cached_constants: HashMap<VestaField, KimchiCellVar>,

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

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // Note: I don't think we need this, but it acts as a nice redundant failsafe.
    pub(crate) finalized: bool,

    /// Size of the public input.
    pub(crate) public_input_size: usize,

    /// Indexes used by the private inputs
    /// (this is useful to check that they appear in the circuit)
    pub(crate) private_input_cell_vars: Vec<KimchiCellVar>,
}

impl Witness {
    /// kimchi uses a transposed witness
    #[must_use]
    pub fn to_kimchi_witness(&self) -> [Vec<VestaField>; NUM_REGISTERS] {
        let transposed = (0..NUM_REGISTERS)
            .map(|_| Vec::with_capacity(self.0.len()))
            .collect::<Vec<_>>();
        let mut transposed: [_; NUM_REGISTERS] = transposed.try_into().unwrap();
        for row in &self.0 {
            for (col, field) in row.iter().enumerate() {
                transposed[col].push(*field);
            }
        }
        transposed
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn debug(&self) {
        for (row, values) in self.0.iter().enumerate() {
            let values = values
                .iter()
                .map(super::super::helpers::PrettyField::pretty)
                .join(" | ");
            println!("{row} - {values}");
        }
    }
}

impl KimchiVesta {
    #[must_use]
    pub fn new(double_generic_gate_optimization: bool) -> Self {
        Self {
            next_variable: 0,
            vars_to_value: HashMap::new(),
            witness_table: vec![],
            cached_constants: HashMap::new(),
            gates: vec![],
            wiring: HashMap::new(),
            double_generic_gate_optimization,
            pending_generic_gate: None,
            debug_info: vec![],
            finalized: false,
            public_input_size: 0,
            private_input_cell_vars: vec![],
        }
    }

    /// Add a gate to the circuit
    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<KimchiCellVar>>,
        coeffs: Vec<VestaField>,
        span: Span,
    ) {
        // sanitize
        assert!(coeffs.len() <= NUM_REGISTERS);
        assert!(vars.len() <= NUM_REGISTERS);

        // construct the execution trace with vars, for the witness generation
        self.witness_table.push(vars.clone());

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
                            *w = Wiring::Wired(vec![old_cell.clone(), annotated_cell.clone()]);
                        }
                        Wiring::Wired(ref mut cells) => {
                            cells.push(annotated_cell.clone());
                        }
                    })
                    .or_insert(Wiring::NotWired(annotated_cell));
            }
        }
    }

    /// Add a generic double gate to the circuit
    fn add_generic_gate(
        &mut self,
        label: &'static str,
        mut vars: Vec<Option<KimchiCellVar>>,
        mut coeffs: Vec<VestaField>,
        span: Span,
    ) {
        // padding
        let coeffs_padding = GENERIC_COEFFS.checked_sub(coeffs.len()).unwrap();
        coeffs.extend(std::iter::repeat(VestaField::zero()).take(coeffs_padding));

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

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KimchiCellVar {
    index: usize,
    pub span: Span,
}

impl BackendVar for KimchiCellVar {}

impl KimchiCellVar {
    fn new(index: usize, span: Span) -> Self {
        Self { index, span }
    }
}

impl Backend for KimchiVesta {
    type Field = VestaField;
    type Var = KimchiCellVar;
    type GeneratedWitness = GeneratedWitness;

    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon
    }

    fn new_internal_var(&mut self, val: Value<KimchiVesta>, span: Span) -> KimchiCellVar {
        // create new var
        let var = KimchiCellVar::new(self.next_variable, span);
        self.next_variable += 1;

        // store it in the circuit_writer
        self.vars_to_value.insert(var.index, val);

        var
    }

    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: VestaField,
        span: Span,
    ) -> KimchiCellVar {
        if let Some(cvar) = self.cached_constants.get(&value) {
            return *cvar;
        }

        let var = self.new_internal_var(Value::Constant(value), span);
        self.cached_constants.insert(value, var);

        let zero = VestaField::zero();

        let () = &self.add_generic_gate(
            label.unwrap_or("hardcode a constant"),
            vec![Some(var)],
            vec![VestaField::one(), zero, zero, zero, value.neg()],
            span,
        );

        var
    }

    #[allow(unused_variables)]
    fn finalize_circuit(
        &mut self,
        public_output: Option<Var<Self::Field, Self::Var>>,
        returned_cells: Option<Vec<KimchiCellVar>>,
        main_span: Span,
    ) -> Result<()> {
        // TODO: the current tests pass even this is commented out. Add a test case for this one.
        // important: there might still be a pending generic gate
        if let Some(pending) = self.pending_generic_gate.take() {
            self.add_gate(
                pending.label,
                GateKind::DoubleGeneric,
                pending.vars,
                pending.coeffs,
                pending.span,
            );
        }

        // for sanity check, we make sure that every cellvar created has ended up in a gate
        let mut written_vars = HashSet::new();
        for row in &self.witness_table {
            row.iter().flatten().for_each(|cvar| {
                written_vars.insert(cvar.index);
            });
        }

        for var in 0..self.next_variable {
            if !written_vars.contains(&var) {
                if let Some(private_cell_var) = self
                    .private_input_cell_vars
                    .iter()
                    .find(|private_cell_var| private_cell_var.index == var)
                {
                    // TODO: is this error useful?
                    let err = Error::new(
                        "constraint-finalization",
                        ErrorKind::PrivateInputNotUsed,
                        private_cell_var.span,
                    );
                    return Err(err);
                } else {
                    panic!("there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!");
                }
            }
        }

        // kimchi hack
        assert!(
            self.gates.len() > 2,
            "the circuit is either too small or does not constrain anything (TODO: better error)"
        );

        // store the return value in the public input that was created for that ^
        if let Some(public_output) = public_output {
            let cvars = &public_output.cvars;

            for (pub_var, ret_var) in cvars.clone().iter().zip(returned_cells.unwrap()) {
                // replace the computation of the public output vars with the actual variables being returned here
                let var_idx = pub_var.cvar().unwrap().index;
                let prev = self
                    .vars_to_value
                    .insert(var_idx, Value::PublicOutput(Some(ret_var)));
                assert!(prev.is_some());
            }
        }

        self.finalized = true;

        Ok(())
    }

    fn compute_var(
        &self,
        env: &mut crate::witness::WitnessEnv<Self::Field>,
        var: &Self::Var,
    ) -> crate::error::Result<Self::Field> {
        let val = self.vars_to_value.get(&var.index).unwrap();
        self.compute_val(env, val, var.index)
    }

    fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<VestaField>,
    ) -> Result<GeneratedWitness> {
        if !self.finalized {
            unreachable!("the circuit must be finalized before generating a witness");
        }

        let mut witness = vec![];
        // compute each rows' vars, except for the deferred ones (public output)
        let mut public_outputs_vars: HashMap<KimchiCellVar, Vec<(usize, usize)>> = HashMap::new();

        // calculate witness except for public outputs
        for (row, row_of_vars) in self.witness_table.iter().enumerate() {
            // create the witness row
            let mut witness_row = [Self::Field::zero(); NUM_REGISTERS];

            for (col, var) in row_of_vars.iter().enumerate() {
                let val = if let Some(var) = var {
                    // if it's a public output, defer it's computation
                    if matches!(
                        self.vars_to_value.get(&var.index),
                        Some(Value::PublicOutput(_))
                    ) {
                        public_outputs_vars
                            .entry(*var)
                            .or_default()
                            .push((row, col));
                        Self::Field::zero()
                    } else {
                        self.compute_var(witness_env, var)?
                    }
                } else {
                    Self::Field::zero()
                };
                witness_row[col] = val;
            }

            witness.push(witness_row);
        }

        // compute public output at last
        let mut public_outputs = vec![];

        for (var, rows_cols) in public_outputs_vars {
            let val = self.compute_var(witness_env, &var)?;
            for (row, col) in rows_cols {
                witness[row][col] = val;
            }
            public_outputs.push(val);
        }

        // sanity check the witness
        for (row, (gate, witness_row, debug_info)) in
            izip!(self.gates.iter(), &witness, &self.debug_info).enumerate()
        {
            let is_not_public_input = row >= self.public_input_size;
            if is_not_public_input {
                #[allow(clippy::single_match)]
                match gate.typ {
                    // only check the generic gate
                    crate::circuit_writer::GateKind::DoubleGeneric => {
                        let c = |i| {
                            gate.coeffs
                                .get(i)
                                .copied()
                                .unwrap_or_else(Self::Field::zero)
                        };
                        let w = &witness_row;
                        let sum1 =
                            c(0) * w[0] + c(1) * w[1] + c(2) * w[2] + c(3) * w[0] * w[1] + c(4);
                        let sum2 =
                            c(5) * w[3] + c(6) * w[4] + c(7) * w[5] + c(8) * w[3] * w[4] + c(9);
                        if sum1 != Self::Field::zero() || sum2 != Self::Field::zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::InvalidWitness(row),
                                debug_info.span,
                            ));
                        }
                    }
                    // for all other gates, we trust the gadgets
                    _ => (),
                }
            }
        }

        // extract full public input (containing the public output)
        let mut full_public_inputs = Vec::with_capacity(self.public_input_size);

        for witness_row in witness.iter().take(self.public_input_size) {
            full_public_inputs.push(witness_row[0]);
        }

        // sanity checks
        assert_eq!(witness.len(), self.gates.len());
        assert_eq!(witness.len(), self.witness_table.len());

        // return the public output separately as well
        Ok(GeneratedWitness {
            all_witness: Witness(witness),
            full_public_inputs,
            public_outputs,
        })
    }

    fn generate_asm(&self, sources: &Sources, debug: bool) -> String {
        let mut res = String::new();

        // version
        res.push_str(&crate::utils::noname_version());

        // vars
        let mut vars: OrderedHashSet<VestaField> = OrderedHashSet::default();

        for Gate { coeffs, .. } in &self.gates {
            extract_vars_from_coeffs(&mut vars, coeffs);
        }

        if debug && !vars.is_empty() {
            crate::utils::title(&mut res, "VARS");
        }

        for (idx, var) in vars.iter().enumerate() {
            writeln!(res, "c{idx} = {}", var.pretty()).unwrap();
        }

        // gates
        if debug {
            crate::utils::title(&mut res, "GATES");
        }

        for (row, (Gate { typ, coeffs }, debug_info)) in
            self.gates.iter().zip(&self.debug_info).enumerate()
        {
            println!("gate {row:?}");
            // gate #
            if debug {
                writeln!(res, "╭{s}", s = "─".repeat(80)).unwrap();
                write!(res, "│ GATE {row} - ").unwrap();
            }

            // gate
            write!(res, "{typ:?}").unwrap();

            // coeffs
            {
                let coeffs = parse_coeffs(&vars, coeffs);
                if !coeffs.is_empty() {
                    res.push('<');
                    res.push_str(&coeffs.join(","));
                    res.push('>');
                }
            }

            res.push('\n');

            if debug {
                // source
                crate::utils::display_source(&mut res, sources, &[debug_info.clone()]);

                // note
                res.push_str("    ▲\n");
                writeln!(res, "    ╰── {note}", note = debug_info.note).unwrap();

                //
                res.push_str("\n\n");
            }
        }

        // wiring
        if debug {
            crate::utils::title(&mut res, "WIRING");
        }

        let mut cycles: Vec<_> = self
            .wiring
            .values()
            .map(|w| match w {
                Wiring::NotWired(_) => None,
                Wiring::Wired(annotated_cells) => Some(annotated_cells),
            })
            .filter(Option::is_some)
            .flatten()
            .collect();

        // we must have a deterministic sort for the cycles,
        // otherwise the same circuit might have different representations
        cycles.sort();

        for annotated_cells in cycles {
            let (cells, debug_infos): (Vec<_>, Vec<_>) = annotated_cells
                .iter()
                .map(|AnnotatedCell { cell, debug }| (*cell, debug.clone()))
                .unzip();

            if debug {
                crate::utils::display_source(&mut res, sources, &debug_infos);
            }

            let s = cells.iter().map(|cell| format!("{cell}")).join(" -> ");
            writeln!(res, "{s}").unwrap();

            if debug {
                writeln!(res, "\n").unwrap();
            }
        }

        res
    }

    fn neg(&mut self, var: &KimchiCellVar, span: Span) -> KimchiCellVar {
        let zero = Self::Field::zero();
        let one = Self::Field::one();

        let neg_var = self.new_internal_var(
            Value::LinearCombination(vec![(one.neg(), *var)], zero),
            span,
        );
        self.add_generic_gate(
            "constraint to validate a negation (`x + (-x) = 0`)",
            vec![Some(*var), Some(neg_var)],
            vec![one, one],
            span,
        );

        neg_var
    }

    fn add(&mut self, lhs: &KimchiCellVar, rhs: &KimchiCellVar, span: Span) -> KimchiCellVar {
        let zero = Self::Field::zero();
        let one = Self::Field::one();

        // create a new variable to store the result
        let res = self.new_internal_var(
            Value::LinearCombination(vec![(one, *lhs), (one, *rhs)], zero),
            span,
        );

        // create a gate to store the result
        self.add_generic_gate(
            "add two variables together",
            vec![Some(*lhs), Some(*rhs), Some(res)],
            vec![one, one, one.neg()],
            span,
        );

        res
    }

    fn add_const(&mut self, var: &KimchiCellVar, cst: &Self::Field, span: Span) -> KimchiCellVar {
        let zero = Self::Field::zero();
        let one = Self::Field::one();

        // create a new variable to store the result
        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *var)], *cst), span);

        // create a gate to store the result
        // TODO: we should use an add_generic function that takes advantage of the double generic gate
        self.add_generic_gate(
            "add a constant with a variable",
            vec![Some(*var), None, Some(res)],
            vec![one, zero, one.neg(), zero, *cst],
            span,
        );

        res
    }

    fn mul(&mut self, lhs: &KimchiCellVar, rhs: &KimchiCellVar, span: Span) -> KimchiCellVar {
        let zero = Self::Field::zero();
        let one = Self::Field::one();

        // create a new variable to store the result
        let res = self.new_internal_var(Value::Mul(*lhs, *rhs), span);

        // create a gate to store the result
        self.add_generic_gate(
            "add two variables together",
            vec![Some(*lhs), Some(*rhs), Some(res)],
            vec![zero, zero, one.neg(), one],
            span,
        );

        res
    }

    fn mul_const(&mut self, var: &KimchiCellVar, cst: &Self::Field, span: Span) -> KimchiCellVar {
        let zero = Self::Field::zero();
        let one = Self::Field::one();

        // create a new variable to store the result
        let res = self.new_internal_var(Value::Scale(*cst, *var), span);

        // create a gate to store the result
        // TODO: we should use an add_generic function that takes advantage of the double generic gate
        self.add_generic_gate(
            "add a constant with a variable",
            vec![Some(*var), None, Some(res)],
            vec![*cst, zero, one.neg()],
            span,
        );

        res
    }

    fn assert_eq_const(&mut self, cvar: &KimchiCellVar, cst: Self::Field, span: Span) {
        self.add_generic_gate(
            "constrain var - cst = 0 to check equality",
            vec![Some(*cvar)],
            vec![
                Self::Field::one(),
                Self::Field::zero(),
                Self::Field::zero(),
                Self::Field::zero(),
                cst.neg(),
            ],
            span,
        );
    }

    fn assert_eq_var(&mut self, lhs: &KimchiCellVar, rhs: &KimchiCellVar, span: Span) {
        // TODO: use permutation to check that
        self.add_generic_gate(
            "constrain lhs - rhs = 0 to assert that they are equal",
            vec![Some(*lhs), Some(*rhs)],
            vec![Self::Field::one(), Self::Field::one().neg()],
            span,
        );
    }

    fn add_public_input(&mut self, val: Value<Self>, span: Span) -> KimchiCellVar {
        // create the var
        let cvar = self.new_internal_var(val, span);

        // create the associated generic gate
        self.add_gate(
            "add public input",
            GateKind::DoubleGeneric,
            vec![Some(cvar)],
            vec![Self::Field::one()],
            span,
        );

        self.public_input_size += 1;

        cvar
    }

    fn add_private_input(&mut self, val: Value<Self>, span: Span) -> Self::Var {
        let cvar = self.new_internal_var(val, span);
        self.private_input_cell_vars.push(cvar);

        cvar
    }

    fn add_public_output(&mut self, val: Value<Self>, span: Span) -> KimchiCellVar {
        // create the var
        let cvar = self.new_internal_var(val, span);

        // create the associated generic gate
        self.add_generic_gate(
            "add public output",
            vec![Some(cvar)],
            vec![Self::Field::one()],
            span,
        );

        self.public_input_size += 1;

        cvar
    }
}

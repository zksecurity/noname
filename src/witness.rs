use std::{collections::HashMap, ops::Deref};

use ark_ff::{Field as _, Zero};
use itertools::{chain, Itertools};

use crate::{
    boolean,
    circuit_writer::{CircuitWriter, Gate, GlobalEnv},
    constants::{Field, Span, NUM_REGISTERS},
    error::{Error, ErrorKind, Result},
    helpers::PrettyField as _,
    inputs::{parse_single_input, JsonInputs},
    parser::TyKind,
    type_checker::TypeChecker,
    var::{CellVar, Value},
};

#[derive(Debug, Default)]
pub struct WitnessEnv {
    pub var_values: HashMap<String, Vec<Field>>,

    pub cached_values: HashMap<CellVar, Field>,
}

impl WitnessEnv {
    pub fn add_value(&mut self, name: String, val: Vec<Field>) {
        assert!(self.var_values.insert(name, val).is_none());
    }

    pub fn get_external(&self, name: &str) -> Vec<Field> {
        // TODO: return an error instead of crashing
        self.var_values.get(name).unwrap().clone()
    }
}

pub struct Witness(Vec<[Field; NUM_REGISTERS]>);

impl Witness {
    /// kimchi uses a transposed witness
    pub fn to_kimchi_witness(&self) -> [Vec<Field>; NUM_REGISTERS] {
        let transposed = vec![Vec::with_capacity(self.0.len()); NUM_REGISTERS];
        let mut transposed: [_; NUM_REGISTERS] = transposed.try_into().unwrap();
        for row in &self.0 {
            for (col, field) in row.iter().enumerate() {
                transposed[col].push(*field);
            }
        }
        transposed
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn debug(&self) {
        for (row, values) in self.0.iter().enumerate() {
            let values = values.iter().map(|v| v.pretty()).join(" | ");
            println!("{row} - {values}");
        }
    }
}

/// The compiled circuit.
pub struct CompiledCircuit {
    pub circuit: CircuitWriter,
    env: GlobalEnv,
}

impl CompiledCircuit {
    pub(crate) fn new(circuit: CircuitWriter, env: GlobalEnv) -> Self {
        Self { circuit, env }
    }

    pub fn asm(&self, debug: bool) -> String {
        self.circuit.asm(debug)
    }

    pub fn compiled_gates(&self) -> &[Gate] {
        self.circuit.compiled_gates()
    }

    pub fn compute_var(&self, env: &mut WitnessEnv, var: CellVar) -> Result<Field> {
        // fetch cache first
        // TODO: if self was &mut, then we could use a Value::Cached(Field) to store things instead of that
        if let Some(res) = env.cached_values.get(&var) {
            return Ok(*res);
        }

        match &self.circuit.witness_vars[&var.index] {
            Value::Hint(func) => {
                let res = func(self, env)
                    .expect("that function doesn't return a var (type checker error)");
                env.cached_values.insert(var, res);
                Ok(res)
            }
            Value::Constant(c) => Ok(*c),
            Value::LinearCombination(lc, cst) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_var(env, *var)? * *coeff;
                }
                env.cached_values.insert(var, res); // cache
                Ok(res)
            }
            Value::Mul(lhs, rhs) => {
                let lhs = self.compute_var(env, *lhs)?;
                let rhs = self.compute_var(env, *rhs)?;
                let res = lhs * rhs;
                env.cached_values.insert(var, res); // cache
                Ok(res)
            }
            Value::Inverse(v) => {
                let v = self.compute_var(env, *v)?;
                let res = v.inverse().unwrap_or_else(Field::zero);
                env.cached_values.insert(var, res); // cache
                Ok(res)
            }
            Value::External(name, idx) => Ok(env.get_external(name)[*idx]),
            Value::PublicOutput(var) => {
                let var = var.ok_or_else(|| Error::new(ErrorKind::MissingReturn, Span(0, 0)))?;
                self.compute_var(env, var)
            }
            Value::Scale(scalar, var) => {
                let var = self.compute_var(env, *var)?;
                Ok(*scalar * var)
            }
        }
    }

    pub fn generate_witness(
        &self,
        mut public_inputs: JsonInputs,
        mut private_inputs: JsonInputs,
    ) -> Result<(Witness, Vec<Field>, Vec<Field>)> {
        let mut witness = vec![];
        let mut env = WitnessEnv::default();

        // create the argument's variables?
        for arg in &self.circuit.main.0.arguments {
            let name = &arg.name.value;

            let input = if arg.is_public() {
                public_inputs.0.remove(name).ok_or_else(|| {
                    Error::new(ErrorKind::MissingPublicArg(name.clone()), arg.span)
                })?
            } else {
                private_inputs.0.remove(name).ok_or_else(|| {
                    Error::new(ErrorKind::MissingPrivateArg(name.clone()), arg.span)
                })?
            };

            let fields = parse_single_input(&self.env, input, &arg.typ.kind)
                .map_err(|e| Error::new(ErrorKind::ParsingError(e), arg.span))?;

            env.add_value(name.clone(), fields.clone());
        }

        // ensure that we've used all of the inputs provided
        if let Some(name) = chain![private_inputs.0.keys(), public_inputs.0.keys()].next() {
            return Err(Error::new(
                ErrorKind::UnusedInput(name.clone()),
                self.circuit.main.1,
            ));
        }

        // compute each rows' vars, except for the deferred ones (public output)
        let mut public_outputs_vars: Vec<(usize, CellVar)> = vec![];

        for (row, (row_of_vars, gate)) in self
            .circuit
            .rows_of_vars
            .iter()
            .zip_eq(self.circuit.compiled_gates())
            .enumerate()
        {
            // create the witness row
            let mut witness_row = [Field::zero(); NUM_REGISTERS];
            for (col, var) in row_of_vars.iter().enumerate() {
                let val = if let Some(var) = var {
                    // if it's a public output, defer it's computation
                    if matches!(
                        self.circuit.witness_vars[&var.index],
                        Value::PublicOutput(_)
                    ) {
                        public_outputs_vars.push((row, *var));
                        Field::zero()
                    } else {
                        self.compute_var(&mut env, *var)?
                    }
                } else {
                    Field::zero()
                };
                witness_row[col] = val;
            }

            // check if the row makes sense
            let is_not_public_input = row >= self.circuit.public_input_size;
            if is_not_public_input {
                #[allow(clippy::single_match)]
                match gate.typ {
                    // only check the generic gate
                    crate::circuit_writer::GateKind::DoubleGeneric => {
                        let c = |i| gate.coeffs.get(i).copied().unwrap_or_else(Field::zero);
                        let w = &witness_row;
                        let sum =
                            c(0) * w[0] + c(1) * w[1] + c(2) * w[2] + c(3) * w[0] * w[1] + c(4);
                        if sum != Field::zero() {
                            return Err(Error::new(ErrorKind::InvalidWitness(row), gate.span));
                        }
                    }
                    // for all other gates, we trust the gadgets
                    _ => (),
                }
            }

            //
            witness.push(witness_row);
        }

        // compute public output at last
        let mut public_output = vec![];

        for (row, var) in public_outputs_vars {
            let val = self.compute_var(&mut env, var)?;
            witness[row][0] = val;
            public_output.push(val);
        }

        // extract full public input (containing the public output)
        let mut full_public_inputs = Vec::with_capacity(self.circuit.public_input_size);

        for witness_row in witness.iter().take(self.circuit.public_input_size) {
            full_public_inputs.push(witness_row[0]);
        }

        // sanity checks
        assert_eq!(witness.len(), self.circuit.num_gates());
        assert_eq!(witness.len(), self.circuit.rows_of_vars.len());

        // return the public output separately as well
        Ok((Witness(witness), full_public_inputs, public_output))
    }
}

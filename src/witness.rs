use std::collections::HashMap;

use ark_ff::Zero;
use itertools::{chain, Itertools};

use crate::{
    ast::{CellValues, CellVar, Compiler, Value},
    constants::NUM_REGISTERS,
    error::{Error, ErrorKind, Result},
    field::{Field, PrettyField},
    inputs::Inputs,
    parser::TyKind,
};

#[derive(Debug, Default)]
pub struct WitnessEnv {
    pub var_values: HashMap<String, CellValues>,

    pub cached_values: HashMap<CellVar, Field>,
}

impl WitnessEnv {
    pub fn add_value(&mut self, name: String, val: CellValues) {
        assert!(self.var_values.insert(name, val).is_none());
    }

    pub fn get_external(&self, name: &str) -> Vec<Field> {
        // TODO: return an error instead of crashing
        self.var_values.get(name).unwrap().clone().values.clone()
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

    pub fn debug(&self) {
        for (row, values) in self.0.iter().enumerate() {
            let values = values.iter().map(|v| v.pretty()).join(" | ");
            println!("{row} - {values}");
        }
    }
}

impl Compiler {
    pub fn compute_var(&self, env: &mut WitnessEnv, var: CellVar) -> Result<Field> {
        // fetch cache first
        // TODO: if self was &mut, then we could use a Value::Cached(Field) to store things instead of that
        if let Some(res) = env.cached_values.get(&var) {
            return Ok(*res);
        }

        match &self.witness_vars[&var] {
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
            Value::External(name, idx) => Ok(env.get_external(name)[*idx]),
            Value::PublicOutput(var) => {
                let var = var.ok_or(Error {
                    kind: ErrorKind::MissingPublicOutput,
                    span: (0, 0),
                })?;
                self.compute_var(env, var)
            }
        }
    }

    pub fn generate_witness(
        &self,
        mut public_inputs: Inputs,
        mut private_inputs: Inputs,
    ) -> Result<(Witness, Vec<Field>, Vec<Field>)> {
        let mut witness = vec![];
        let mut env = WitnessEnv::default();

        // create the argument's variables?
        for (name, arg) in &self.main_args.0 {
            let cval = match &arg.typ.kind {
                TyKind::Field => {
                    let input = if arg.is_public() {
                        public_inputs.0.remove(name)
                    } else {
                        private_inputs.0.remove(name)
                    };

                    input.ok_or(Error {
                        kind: ErrorKind::MissingArg(name.clone()),
                        span: arg.span,
                    })?
                }
                TyKind::Array(array_typ, size) if **array_typ == TyKind::Field => {
                    let input = if arg.is_public() {
                        public_inputs.0.remove(name)
                    } else {
                        private_inputs.0.remove(name)
                    };

                    let cval = input.ok_or(Error {
                        kind: ErrorKind::MissingArg(name.clone()),
                        span: arg.span,
                    })?;
                    if cval.values.len() != *size as usize {
                        panic!("convert this to an error");
                    }
                    cval
                }
                _ => unimplemented!(),
            };

            env.add_value(name.clone(), cval.clone());
        }

        // ensure that we've used all of the inputs provided
        for name in chain![private_inputs.0.keys(), public_inputs.0.keys()] {
            return Err(Error {
                kind: ErrorKind::UnusedInput(name.clone()),
                span: self.main_args.1,
            });
        }

        // compute each rows' vars, except for the deferred ones (public output)
        let mut public_outputs_vars: Vec<(usize, CellVar)> = vec![];

        for (row, (row_of_vars, gate)) in self
            .rows_of_vars
            .iter()
            .zip_eq(self.compiled_gates())
            .enumerate()
        {
            // create the witness row
            let mut witness_row = [Field::zero(); NUM_REGISTERS];
            for (col, var) in row_of_vars.iter().enumerate() {
                let val = if let Some(var) = var {
                    // if it's a public output, defer it's computation
                    if matches!(self.witness_vars[&var], Value::PublicOutput(_)) {
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
            let is_not_public_input = row >= self.public_input_size;
            if is_not_public_input {
                match gate.typ {
                    // only check the generic gate
                    crate::ast::GateKind::DoubleGeneric => {
                        let c = |i| gate.coeffs.get(i).copied().unwrap_or(Field::zero());
                        let w = &witness_row;
                        let sum =
                            c(0) * w[0] + c(1) * w[1] + c(2) * w[2] + c(3) * w[0] * w[1] + c(4);
                        if sum != Field::zero() {
                            dbg!(format!("{}", w[0]));
                            dbg!(format!("{}", w[1]));
                            return Err(Error {
                                kind: ErrorKind::InvalidWitness(row),
                                span: gate.span,
                            });
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
        let mut full_public_inputs = Vec::with_capacity(self.public_input_size);
        for row in 0..self.public_input_size {
            full_public_inputs.push(witness[row][0]);
        }

        //
        assert_eq!(witness.len(), self.num_gates());

        Ok((Witness(witness), full_public_inputs, public_output))
    }
}

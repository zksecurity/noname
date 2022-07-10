use std::collections::HashMap;

use ark_ff::Zero;
use itertools::Itertools;

use crate::{
    ast::{CellValues, CellVar, Compiler, Value},
    constants::IO_REGISTERS,
    error::{Error, ErrorKind, Result},
    field::{Field, PrettyField},
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

pub struct Witness(Vec<[Field; IO_REGISTERS]>);

impl Witness {
    /// kimchi uses a transposed witness
    pub fn to_kimchi_witness(&self) -> [Vec<Field>; IO_REGISTERS] {
        let transposed = vec![Vec::with_capacity(self.0.len()); IO_REGISTERS];
        let mut transposed: [_; IO_REGISTERS] = transposed.try_into().unwrap();
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
        args: HashMap<&str, CellValues>,
    ) -> Result<(Witness, Vec<Field>, Vec<Field>)> {
        let mut witness = vec![];
        let mut env = WitnessEnv::default();

        // create the argument's variables?
        for (name, (typ, span)) in &self.main_args {
            let cval = match typ {
                TyKind::Field => args.get(name.as_str()).ok_or(Error {
                    kind: ErrorKind::MissingArg(name.clone()),
                    span: *span,
                })?,
                TyKind::Array(array_typ, size) if **array_typ == TyKind::Field => {
                    let cval = args.get(name.as_str()).ok_or(Error {
                        kind: ErrorKind::MissingArg(name.clone()),
                        span: *span,
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

        // compute each rows' vars, except for the deferred ones (public output)
        let mut public_outputs_vars: Vec<(usize, CellVar)> = vec![];

        for (row, witness_row) in self.witness_rows.iter().enumerate() {
            // create the witness row
            let mut res = [Field::zero(); IO_REGISTERS];
            for (col, var) in witness_row.iter().enumerate() {
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
                res[col] = val;
            }

            //
            witness.push(res);
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

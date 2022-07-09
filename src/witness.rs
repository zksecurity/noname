use std::collections::HashMap;

use ark_ff::Zero;
use itertools::Itertools;

use crate::{
    ast::{CellVar, CircuitValue, Compiler, Value},
    constants::IO_REGISTERS,
    error::{Error, ErrorKind},
    field::{Field, PrettyField},
    parser::TyKind,
};

#[derive(Debug, Default)]
pub struct WitnessEnv {
    pub var_values: HashMap<String, CircuitValue>,
}

impl WitnessEnv {
    pub fn add_value(&mut self, name: String, val: CircuitValue) {
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
    pub fn compute_var(&self, env: &WitnessEnv, var: CellVar) -> Result<Field, Error> {
        match &self.witness_vars[&var] {
            Value::Hint(func) => func(self, env),
            Value::Constant(c) => Ok(*c),
            Value::LinearCombination(lc, cst) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_var(env, *var)? * *coeff;
                }
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
        args: HashMap<&str, CircuitValue>,
    ) -> Result<(Witness, Vec<Field>), Error> {
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
                        self.compute_var(&env, *var)?
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
            let val = self.compute_var(&env, var)?;
            witness[row][0] = val;
            public_output.push(val);
        }

        //
        assert_eq!(witness.len(), self.num_gates());

        Ok((Witness(witness), public_output))
    }
}

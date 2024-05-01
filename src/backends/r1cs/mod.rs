pub mod builtin;
pub mod snarkjs;

use std::collections::{HashMap, HashSet};

use std::ops::Neg;
use ark_bls12_381::Fr;
use ark_ff::{Field, Zero};

use num_bigint_dig::{BigInt, Sign};

use crate::{circuit_writer::DebugInfo, var::{CellVar, Value}};

use self::builtin::poseidon;

use super::Backend;

/// Linear combination of variables and constants
/// For example, the linear combination is represented as a * f_a + b * f_b + c
/// f_a and f_b are the coefficients of a and b respectively.
/// a and b are represented by CellVar
/// the constant c is represented by the constant field, will always multiply with the variable at index 0 which is always 1
#[derive(Clone, Debug)]
pub struct LinearCombination {
    pub terms: Option<HashMap<CellVar, Fr>>,
    pub constant: Option<Fr>,
}

impl LinearCombination {
    /// Evaluate the linear combination with the given witness
    pub fn evaluate(&self, witness: &HashMap<usize, Fr>) -> Fr {
        let mut sum = Fr::zero();

        if let Some(terms) = &self.terms {
            for (var, factor) in terms {
                sum += *witness.get(&var.index).unwrap() * factor;
            }
        }

        if let Some(constant) = &self.constant {
            sum += *constant;
        }

        sum
    }
}

/// a R1CS constraint
/// Each constraint comprises of 3 linear combinations from 3 matrixes 
/// It represents a constraint in math: a * b = c
#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
}

impl Constraint {
    /// Convert the 3 linear combinations to an array 
    pub fn as_array(&self) -> [&LinearCombination; 3] {
        [&self.a, &self.b, &self.c]
    }
}


/// R1CS backend with bls12_381 field
#[derive(Clone)]
pub struct R1csBls12_381 {
    /// so that we can get the associated field type, instead of passing it as a parameter here.
    /// there are two fields supported by the snarkjs for r1cs: bn128 and ark_bls12_381.
    /// current we are using ark_bls12_381
    // todo: how can we create this r1cs backend with different associated field types, but still using the same backend implementation? using macro?
    prime: BigInt,
    /// constraints in the r1cs
    constraints: Vec<Constraint>,
    next_variable: usize,
    witness_vars: HashMap<usize, Value<R1csBls12_381>>,
    debug_info: Vec<DebugInfo>,
    /// record the public inputs for reordering the witness vector
    public_inputs: Vec<CellVar>,
    /// record the public outputs for reordering the witness vector
    public_outputs: Vec<CellVar>,
    finalized: bool,
}

impl R1csBls12_381 {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            next_variable: 0,
            witness_vars: HashMap::new(),
            debug_info: Vec::new(),
            public_inputs: Vec::new(),
            public_outputs: Vec::new(),
            prime: Self::bigint_from_u64_slice(Fr::characteristic()),
            finalized: false,
        }
    }

    /// Add a r1cs constraint that is 3 linear combinations.
    /// This represents one constraint: a * b = c
    pub fn add_constraint(
        &mut self,
        note: &str,
        c: Constraint,
        span: crate::constants::Span,
    ) {
        let debug_info = DebugInfo {
            note: note.to_string(),
            span,
        };
        self.debug_info.push(debug_info);

        self.constraints.push(c);
    }

    /// Compute the number of private inputs 
    /// based on the number of all witness variables, public inputs and public outputs
    pub fn private_input_number(&self) -> usize {
        self.witness_vars.len() - self.public_inputs.len() - self.public_outputs.len()
    }

    // Helper function to convert &[u64] to BigInt
    fn bigint_from_u64_slice(slice: &[u64]) -> BigInt {
        let mut bytes = Vec::new();
        for &num in slice.iter().rev() { // Reverse the slice to match the big-endian byte order
            bytes.extend(&num.to_be_bytes());
        }
        BigInt::from_bytes_be(Sign::Plus, &bytes)
    }
}

#[derive(Debug)]
/// An intermediate struct for SnarkjsExporter to reorder the witness and convert to snarkjs format 
pub struct GeneratedWitness {
    pub witness: HashMap<usize, Fr>,
}

impl Backend for R1csBls12_381 {
    type Field = Fr;

    type GeneratedWitness = GeneratedWitness;

    fn witness_vars(&self) -> &std::collections::HashMap<usize, crate::var::Value<Self>> {
        &self.witness_vars
    }

    fn poseidon() -> crate::imports::FnHandle<Self> {
        poseidon
    }

    fn new_internal_var(
        &mut self,
        val: crate::var::Value<Self>,
        span: crate::constants::Span,
    ) -> CellVar {
        let var = CellVar::new(self.next_variable, span);
        self.next_variable += 1;

        // store it in the circuit_writer
        self.witness_vars.insert(var.index, val);

        var
    }

    fn add_constant(
        &mut self,
        //todo: do we need this?
        label: Option<&'static str>,
        value: Self::Field,
        span: crate::constants::Span,
    ) -> CellVar {
        let x = self.new_internal_var(Value::Constant(value), span);
        self.assert_eq_const(&x, value, span);

        x
    }

    /// Final checks for generating the circuit
    /// todo: we might need to rethink about this interface
    /// - private_input_indices are not needed in this r1cs backend
    /// - main_span could be better initialized with the backend, so it doesn't have to pass in here?
    /// - we might just need the returned_cells argument, as the backend can record the public outputs itself?
    fn finalize_circuit(
        &mut self,
        public_output: Option<crate::var::Var<Self::Field>>,
        returned_cells: Option<Vec<CellVar>>,
        _private_input_indices: Vec<usize>,
        _main_span: crate::constants::Span,
    ) -> crate::error::Result<()> {
        // store the return value in the public input that was created for that
        if let Some(public_output) = public_output {
            let cvars = &public_output.cvars;

            for (pub_var, ret_var) in cvars.clone().iter().zip(returned_cells.unwrap()) {
                // replace the computation of the public output vars with the actual variables being returned here
                let var_idx = pub_var.idx().unwrap();
                let prev = self
                    .witness_vars
                    .insert(var_idx, Value::PublicOutput(Some(ret_var)));
                assert!(prev.is_some());
            }
        }

        // for sanity check, we make sure that every cellvar created has ended up in a r1cs constraint
        let mut written_vars = HashSet::new();
        for constraint in &self.constraints {
            for lc in constraint.as_array() {
                if let Some(terms) = &lc.terms {
                    for var in terms.keys() {
                        written_vars.insert(var.index);
                    }
                }
            }
        }

        // check if every cell vars end up being a cell var in the circuit or public output
        for index in 0..self.next_variable {
            if !written_vars.contains(&index) {
                // check if outputs contains the cell var that has the same index
                if !self.public_outputs.iter().any(|&x| x.index == index) {
                    panic!("there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!");
                }
            }
        }

        self.finalized = true;

        Ok(())
    }

    /// Generate the witnesses 
    /// This process should check if the constraints are satisfied.
    fn generate_witness(
        &self,
        witness_env: &mut crate::witness::WitnessEnv<Self::Field>,
    ) -> crate::error::Result<Self::GeneratedWitness> {
        if !self.finalized {
            panic!("the circuit is not finalized yet!");
        }

        let mut witness = HashMap::<usize, Fr>::new();

        for constraint in &self.constraints {
            for lc in &constraint.as_array() {
                if let Some(terms) = &lc.terms {
                    for var in terms.keys() {
                        if witness.contains_key(&var.index) {
                            continue;
                        }
                        let val = self.compute_var(witness_env, *var)?;
                        witness.insert(var.index, val);
                    }
                }
            }
            // assert a * b = c
            assert_eq!(
                constraint.a.evaluate(&witness) * constraint.b.evaluate(&witness), 
                constraint.c.evaluate(&witness)
            );
        }

        // The original vars of public outputs are not part of the constraints
        // so we need to compute them separately
        for var in &self.public_outputs {
            if witness.contains_key(&var.index) {
                continue;
            }
            let val = self.compute_var(witness_env, *var)?;
            witness.insert(var.index, val);
        }

        Ok(GeneratedWitness { witness })
    }

    // todo: we can think of a format for r1cs for easier debugging
    fn generate_asm(&self, sources: &crate::compiler::Sources, debug: bool) -> String {
        todo!()
    }
    
    /// to constraint:
    /// x + (-x) = 0
    /// given:
    /// a * b = c
    /// then:
    /// a = x + (-x)
    /// b = 1
    /// c = 0
    fn neg(&mut self, x: &CellVar, span: crate::constants::Span) -> CellVar {
        let one = Fr::from(1);
        let zero = Fr::from(0);

        let x_neg = self.new_internal_var(Value::LinearCombination(vec![(one.neg(), *x)], zero), span);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*x, one), (x_neg, one)])),
            constant: None,
        };
        let b = LinearCombination {
            terms: None,
            constant: Some(one),
        };
        let c = LinearCombination {
            terms: None,
            constant: Some(zero),
        };

        self.add_constraint(
            "neg constraint: x + (-x) = 0",
            Constraint{a, b, c},
            span
        );

        x_neg
    }
    
    /// to constraint:
    /// lhs + rhs = res
    /// given:
    /// a * b = c
    /// then:
    /// a = lhs + rhs
    /// b = 1
    /// c = res
    fn add(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) -> CellVar {
        let one = Fr::from(1);
        let zero = Fr::from(0);

        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *lhs), (one, *rhs)], zero), span);

        // IMPORTANT: since terms use CellVar as key,
        // HashMap automatically overrides it to single one term if the two vars are the same CellVar
        let a = if lhs == rhs {
            LinearCombination {
                terms: Some(HashMap::from_iter(vec![(*lhs, Fr::from(2))])),
                constant: None,
            }
        } else {
            LinearCombination {
                terms: Some(HashMap::from_iter(vec![(*lhs, one), (*rhs, one)])),
                constant: None,
            }
        };

        let b = LinearCombination {
            terms: None,
            constant: Some(one),
        };

        let c = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(res, one)])),
            constant: None,
        };

        self.add_constraint(
            "add constraint: lhs + rhs = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    /// to constraint:
    /// x + cst = res
    /// given:
    /// a * b = c
    /// then:
    /// a = x + cst
    /// b = 1
    /// c = res
    fn add_const(&mut self, x: &CellVar, cst: &Self::Field, span: crate::constants::Span) -> CellVar {
        let one = Fr::from(1);

        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *x)], *cst), span);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*x, one)])),
            constant: Some(*cst),
        };

        let b = LinearCombination {
            terms: None,
            constant: Some(one),
        };

        let c = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(res, one)])),
            constant: None,
        };

        self.add_constraint(
            "add constraint: x + cst = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    /// to constraint:
    /// lhs * rhs = res
    /// given:
    /// a * b = c
    /// then:
    /// a = lhs
    /// b = rhs
    /// c = res
    fn mul(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) -> CellVar {
        let one = Fr::from(1);

        let res = self.new_internal_var(Value::Mul(*lhs, *rhs), span);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*lhs, one)])),
            constant: None,
        };

        let b = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*rhs, one)])),
            constant: None,
        };

        let c = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(res, one)])),
            constant: None,
        };

        self.add_constraint(
            "mul constraint: lhs * rhs = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    /// to constraint:
    /// x * cst = res
    /// given:
    /// a * b = c
    /// then:
    /// a = x
    /// b = cst
    /// c = res
    fn mul_const(&mut self, x: &CellVar, cst: &Self::Field, span: crate::constants::Span) -> CellVar {
        let one = Fr::from(1);

        let res = self.new_internal_var(Value::Scale(*cst, *x), span);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*x, one)])),
            constant: None,
        };

        let b = LinearCombination {
            terms: None,
            constant: Some(*cst),
        };

        let c = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(res, one)])),
            constant: None,
        };

        self.add_constraint(
            "mul constraint: x * cst = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    /// to constraint:
    /// x = cst
    /// given:
    /// a * b = c
    /// then:
    /// a = x
    /// b = 1
    /// c = cst
    fn assert_eq_const(&mut self, x: &CellVar, cst: Self::Field, span: crate::constants::Span) {
        let one = Fr::from(1);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*x, one)])),
            constant: None,
        };

        let b = LinearCombination {
            terms: None,
            constant: Some(one),
        };

        let c = LinearCombination {
            terms: None,
            constant: Some(cst),
        };

        self.add_constraint(
            "eq constraint: x = cst",
            Constraint{a, b, c},
            span
        );
    }

    /// to constraint:
    /// lhs = rhs
    /// given:
    /// a * b = c
    /// then:
    /// a = lhs
    /// b = 1
    /// c = rhs
    fn assert_eq_var(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) {
        let one = Fr::from(1);

        let a = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*lhs, one)])),
            constant: None,
        };

        let b = LinearCombination {
            terms: None,
            constant: Some(one),
        };

        let c = LinearCombination {
            terms: Some(HashMap::from_iter(vec![(*rhs, one)])),
            constant: None,
        };

        self.add_constraint(
            "eq constraint: lhs = rhs",
            Constraint{a, b, c},
            span
        );
    }
    
    /// Adds the public input cell vars
    fn add_public_input(&mut self, val: Value<Self>, span: crate::constants::Span) -> CellVar {
        let var = self.new_internal_var(val, span);
        self.public_inputs.push(var);

        var
    }
    
    /// Adds the public output cell vars
    fn add_public_output(&mut self, val: Value<Self>, span: crate::constants::Span) -> CellVar {
        let var = self.new_internal_var(val, span);
        self.public_outputs.push(var);

        var
    }
}


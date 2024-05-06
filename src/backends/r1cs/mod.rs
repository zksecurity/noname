pub mod builtin;
pub mod snarkjs;

use std::collections::{HashMap, HashSet};

use std::ops::Neg;
use ark_bls12_381::Fr;
use ark_ff::{Field, Zero};

use kimchi::circuits::polynomials::foreign_field_add::witness;
use num_bigint_dig::{BigInt, Sign};

use crate::{circuit_writer::DebugInfo, var::{CellVar, Value}};

use self::builtin::poseidon;

use super::Backend;

/// Linear combination of variables and constants.
/// For example, the linear combination is represented as a * f_a + b * f_b + f_c
/// f_a and f_b are the coefficients of a and b respectively.
/// a and b are represented by CellVar.
/// The constant f_c is represented by the constant field, will always multiply with the variable at index 0 which is always 1.
#[derive(Clone, Debug)]
pub struct LinearCombination {
    pub terms: HashMap<CellVar, Fr>,
    pub constant: Fr,
}

impl LinearCombination {
    /// Evaluate the linear combination with the given witness.
    fn evaluate(&self, witness: &[Fr]) -> Fr {
        let mut sum = Fr::zero();

        for (var, factor) in &self.terms {
            sum += *witness.get(var.index).unwrap() * factor;
        }

        sum += &self.constant;

        sum
    }

    /// Create a linear combination to represent constant one.
    fn one() -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: Fr::from(1),
        }
    }

    /// Create a linear combination to represent constant zero.
    fn zero() -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: Fr::from(0),
        }
    }

    /// Create a linear combination from a list of vars
    fn from_vars(vars: Vec<CellVar>) -> Self {
        let terms = vars.into_iter().map(|var| (var, Fr::from(1))).collect();
        LinearCombination {
            terms,
            constant: Fr::from(0),
        }
    }

    /// Create a linear combination from a constant.
    fn from_const(cst: Fr) -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: cst,
        }
    }
}

/// an R1CS constraint
/// Each constraint comprises of 3 linear combinations from 3 matrices.
/// It represents a constraint in math: a * b = c.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
}

impl Constraint {
    /// Convert the 3 linear combinations to an array.
    fn as_array(&self) -> [&LinearCombination; 3] {
        [&self.a, &self.b, &self.c]
    }
}


/// R1CS backend with bls12_381 field.
#[derive(Clone)]
pub struct R1csBls12_381 {
    /// Constraints in the r1cs.
    constraints: Vec<Constraint>,
    witness_vars: Vec<Value<R1csBls12_381>>,
    debug_info: Vec<DebugInfo>,
    /// Record the public inputs for reordering the witness vector
    public_inputs: Vec<CellVar>,
    /// Record the public outputs for reordering the witness vector
    public_outputs: Vec<CellVar>,
    finalized: bool,
}

impl R1csBls12_381 {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            witness_vars: Vec::new(),
            debug_info: Vec::new(),
            public_inputs: Vec::new(),
            public_outputs: Vec::new(),
            finalized: false,
        }
    }

    // todo: how can we create this r1cs backend with different associated field types, but still using the same backend implementation? using macro?
    /// So that we can get the associated field type, instead of passing it as a parameter here.
    /// There are two fields supported by the snarkjs for r1cs: bn128 and ark_bls12_381.
    /// Currently we are using ark_bls12_381
    fn prime(&self) -> BigInt {
        BigInt::from_slice_native(Sign::Plus, Fr::characteristic())
    }

    /// Add an r1cs constraint that is 3 linear combinations.
    /// This represents one constraint: a * b = c
    fn add_constraint(
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
    /// based on the number of all witness variables, public inputs and public outputs.
    fn private_input_number(&self) -> usize {
        self.witness_vars.len() - self.public_inputs.len() - self.public_outputs.len()
    }
}

#[derive(Debug)]
/// An intermediate struct for SnarkjsExporter to reorder the witness and convert to snarkjs format.
pub struct GeneratedWitness {
    pub witness: Vec<Fr>,
}

impl Backend for R1csBls12_381 {
    type Field = Fr;

    type GeneratedWitness = GeneratedWitness;

    fn witness_vars(&self, var: CellVar) -> &Value<Self> {
        self.witness_vars.get(var.index).unwrap()
    }

    fn poseidon() -> crate::imports::FnHandle<Self> {
        poseidon
    }

    fn new_internal_var(
        &mut self,
        val: crate::var::Value<Self>,
        span: crate::constants::Span,
    ) -> CellVar {
        let var = CellVar::new(self.witness_vars.len(), span);

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

    /// Final checks for generating the circuit.
    /// todo: we might need to rethink about this interface
    /// - private_input_indices are not needed in this r1cs backend.
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
                let prev = &self.witness_vars[var_idx];
                assert!(matches!(prev, Value::PublicOutput(None)));
                self.witness_vars[var_idx] = Value::PublicOutput(Some(ret_var));
            }
        }

        // for sanity check, we make sure that every cellvar created has ended up in an r1cs constraint
        let mut written_vars = HashSet::new();
        for constraint in &self.constraints {
            for lc in constraint.as_array() {
                for var in lc.terms.keys() {
                    written_vars.insert(var.index);
                }
            }
        }

        // check if every cell vars end up being a cell var in the circuit or public output
        for index in 0..self.witness_vars.len() {
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

        // generate witness through witness vars vector
        let mut witness = self.witness_vars.iter().enumerate().map(|(index, val)| {
            self.compute_val(witness_env, val, index)
        }).collect::<crate::error::Result<Vec<Fr>>>()?;

        // The original vars of public outputs are not part of the constraints
        // so we need to compute them separately
        for var in &self.public_outputs {
            let val = self.compute_var(witness_env, *var)?;
            witness[var.index] = val;
        }

        for constraint in &self.constraints {
            // assert a * b = c
            assert_eq!(
                constraint.a.evaluate(&witness) * constraint.b.evaluate(&witness), 
                constraint.c.evaluate(&witness)
            );
        }

        Ok(GeneratedWitness { witness })
    }

    // todo: we can think of a format for r1cs for easier debugging
    fn generate_asm(&self, sources: &crate::compiler::Sources, debug: bool) -> String {
        todo!()
    }
    
    fn neg(&mut self, x: &CellVar, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // x + (-x) = 0
        // given:
        // a * b = c
        // then:
        // a = x + (-x)
        // b = 1
        // c = 0
        let one = Fr::from(1);
        let zero = Fr::from(0);

        let x_neg = self.new_internal_var(Value::LinearCombination(vec![(one.neg(), *x)], zero), span);

        let a = LinearCombination::from_vars(vec![*x, x_neg]);
        let b = LinearCombination::one();
        let c = LinearCombination::zero();

        self.add_constraint(
            "neg constraint: x + (-x) = 0",
            Constraint{a, b, c},
            span
        );

        x_neg
    }
    
    fn add(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // lhs + rhs = res
        // given:
        // a * b = c
        // then:
        // a = lhs + rhs
        // b = 1
        // c = res
        let one = Fr::from(1);
        let zero = Fr::from(0);

        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *lhs), (one, *rhs)], zero), span);

        // IMPORTANT: since terms use CellVar as key,
        // HashMap automatically overrides it to single one term if the two vars are the same CellVar
        let a = if lhs == rhs {
            LinearCombination {
                terms: HashMap::from_iter(vec![(*lhs, Fr::from(2))]),
                constant: zero,
            }
        } else {
            LinearCombination::from_vars(vec![*lhs, *rhs])
        };

        let b = LinearCombination::one();
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "add constraint: lhs + rhs = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    fn add_const(&mut self, x: &CellVar, cst: &Self::Field, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // x + cst = res
        // given:
        // a * b = c
        // then:
        // a = x + cst
        // b = 1
        // c = res
        let one = Fr::from(1);

        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *x)], *cst), span);

        let a = LinearCombination {
            terms: HashMap::from_iter(vec![(*x, one)]),
            constant: *cst,
        };

        let b = LinearCombination::one();
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "add constraint: x + cst = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    fn mul(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // lhs * rhs = res
        // given:
        // a * b = c
        // then:
        // a = lhs
        // b = rhs
        // c = res

        let res = self.new_internal_var(Value::Mul(*lhs, *rhs), span);

        let a = LinearCombination::from_vars(vec![*lhs]);
        let b = LinearCombination::from_vars(vec![*rhs]);
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "mul constraint: lhs * rhs = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    fn mul_const(&mut self, x: &CellVar, cst: &Self::Field, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // x * cst = res
        // given:
        // a * b = c
        // then:
        // a = x
        // b = cst
        // c = res

        let res = self.new_internal_var(Value::Scale(*cst, *x), span);

        let a = LinearCombination::from_vars(vec![*x]);
        let b = LinearCombination::from_const(*cst);
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "mul constraint: x * cst = res",
            Constraint{a, b, c},
            span
        );

        res
    }
    
    fn assert_eq_const(&mut self, x: &CellVar, cst: Self::Field, span: crate::constants::Span) {
        // To constrain:
        // x = cst
        // given:
        // a * b = c
        // then:
        // a = x
        // b = 1
        // c = cst

        let a = LinearCombination::from_vars(vec![*x]);
        let b = LinearCombination::one();
        let c = LinearCombination::from_const(cst);

        self.add_constraint(
            "eq constraint: x = cst",
            Constraint{a, b, c},
            span
        );
    }

    fn assert_eq_var(&mut self, lhs: &CellVar, rhs: &CellVar, span: crate::constants::Span) {
        // To constrain:
        // lhs = rhs
        // given:
        // a * b = c
        // then:
        // a = lhs
        // b = 1
        // c = rhs

        let a = LinearCombination::from_vars(vec![*lhs]);
        let b = LinearCombination::one();
        let c = LinearCombination::from_vars(vec![*rhs]);

        self.add_constraint(
            "eq constraint: lhs = rhs",
            Constraint{a, b, c},
            span
        );
    }
    
    /// Adds the public input cell vars.
    fn add_public_input(&mut self, val: Value<Self>, span: crate::constants::Span) -> CellVar {
        let var = self.new_internal_var(val, span);
        self.public_inputs.push(var);

        var
    }
    
    /// Adds the public output cell vars.
    fn add_public_output(&mut self, val: Value<Self>, span: crate::constants::Span) -> CellVar {
        let var = self.new_internal_var(val, span);
        self.public_outputs.push(var);

        var
    }
}


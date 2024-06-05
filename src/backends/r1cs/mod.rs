// TODO: There is a bunch of places where there are unused vars.
// Remove this lint allowance when fixed.
#![allow(unused_variables)]

pub mod builtin;
pub mod snarkjs;

use std::collections::{HashMap, HashSet};

use ark_ff::FpParameters;
use itertools::{izip, Itertools as _};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use crate::constants::Span;
use crate::error::{Error, ErrorKind, Result};
use crate::{circuit_writer::DebugInfo, var::Value};

use super::{Backend, BackendField, BackendVar};

pub type R1csBls12381Field = ark_bls12_381::Fr;
pub type R1csBn254Field = ark_bn254::Fr;

// Because the associated field type is BackendField, we need to implement it for the actual field types in order to use them.
impl BackendField for R1csBls12381Field {}
impl BackendField for R1csBn254Field {}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellVar {
    index: usize,
    pub span: Span,
}

impl CellVar {
    /// Convert to linear combination.
    fn to_linear_combination<F: BackendField>(self) -> LinearCombination<F> {
        LinearCombination::from(self)
    }
}

impl<F: BackendField> BackendVar for LinearCombination<F> {}

/// Linear combination of variables and constants.
/// For example, the linear combination is represented as a * `f_a` + b * `f_b` + `f_c`
/// `f_a` and `f_b` are the coefficients of a and b respectively.
/// a and b are represented by `CellVar`.
/// The constant `f_c` is represented by the constant field, will always multiply with the variable at index 0 which is always 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearCombination<F>
where
    F: BackendField,
{
    pub terms: HashMap<CellVar, F>,
    pub constant: F,
    pub span: Span,
}

impl<F> LinearCombination<F>
where
    F: BackendField,
{
    /// Convert to a `CellVar`.
    /// It should
    /// - be used when the linear combination is a single variable.
    /// - panic if the linear combination is not a single variable or has a non-zero constant.
    /// - panic if the single variable has factor other than 1.
    fn to_cell_var(&self) -> &CellVar {
        assert_eq!(self.terms.len(), 1);
        assert_eq!(self.constant, F::zero());

        let (var, factor) = self.terms.iter().next().unwrap();
        assert_eq!(*factor, F::one());

        var
    }

    /// Evaluate the linear combination with the given witness.
    fn evaluate(&self, witness: &[F]) -> F {
        let mut sum = F::zero();

        for (var, factor) in &self.terms {
            sum += *witness.get(var.index).unwrap() * factor;
        }

        sum += &self.constant;

        sum
    }

    /// Create a linear combination to represent constant one.
    fn one(span: Span) -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: F::one(),
            span,
        }
    }

    /// Create a linear combination from a constant.
    fn from_const(cst: F, span: Span) -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: cst,
            span,
        }
    }

    /// Add with another linear combination.
    fn add(&self, other: &Self, span: Span) -> Self {
        let mut terms = self.terms.clone();

        other.terms.iter().for_each(|(var, c1)| {
            terms.entry(*var).and_modify(|c2| *c2 += c1).or_insert(*c1);
        });

        LinearCombination {
            terms,
            constant: self.constant + other.constant,
            span,
        }
    }

    /// Scale the linear combination with a constant.
    fn scale(&self, coeff: F, span: Span) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(var, factor)| (*var, *factor * coeff))
            .collect();

        LinearCombination {
            terms,
            constant: self.constant * coeff,
            span,
        }
    }

    /// Enforces a constraint for the multiplication of two `CellVars`.
    /// The constraint reduces the multiplication to a new `CellVar` variable,
    /// which represents: self * other = res.
    fn mul(&self, cs: &mut R1CS<F>, other: &Self, span: Span) -> Self {
        let res = cs.new_internal_var(Value::Mul(self.clone(), other.clone()), span);
        cs.enforce_constraint(self, other, &res, span);

        res
    }

    /// Enforces a constraint for the equality of two `CellVars`.
    /// It needs to constraint: self * 1 = other.
    fn assert_eq(&self, cs: &mut R1CS<F>, other: &Self, span: Span) {
        let one_cvar = LinearCombination::one(span);

        cs.enforce_constraint(self, &one_cvar, other, span);
    }
}

impl<F> From<CellVar> for LinearCombination<F>
where
    F: BackendField,
{
    fn from(var: CellVar) -> Self {
        LinearCombination {
            terms: [(var, F::one())].into(),
            constant: F::zero(),
            span: var.span,
        }
    }
}

/// an R1CS constraint
/// Each constraint comprises of 3 linear combinations from 3 matrices.
/// It represents a constraint in math: a * b = c.
#[derive(Clone, Debug)]
pub struct Constraint<F>
where
    F: BackendField,
{
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

impl<F> Constraint<F>
where
    F: BackendField,
{
    /// Convert the 3 linear combinations to an array.
    fn as_array(&self) -> [&LinearCombination<F>; 3] {
        [&self.a, &self.b, &self.c]
    }
}

/// R1CS backend with `bls12_381` field.
#[derive(Clone)]
pub struct R1CS<F>
where
    F: BackendField,
{
    /// Constraints in the r1cs.
    constraints: Vec<Constraint<F>>,
    witness_vector: Vec<Value<Self>>,
    debug_info: Vec<DebugInfo>,
    /// Record the public inputs for reordering the witness vector
    public_inputs: Vec<CellVar>,
    /// Record the private inputs for checking
    private_input_cell_vars: Vec<CellVar>,
    /// Record the public outputs for reordering the witness vector
    public_outputs: Vec<CellVar>,
    finalized: bool,
}

impl<F> Default for R1CS<F>
where
    F: BackendField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> R1CS<F>
where
    F: BackendField,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            witness_vector: Vec::new(),
            debug_info: Vec::new(),
            public_inputs: Vec::new(),
            private_input_cell_vars: Vec::new(),
            public_outputs: Vec::new(),
            finalized: false,
        }
    }

    /// Returns the number of constraints.
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the prime for snarkjs based on the curve field.
    fn prime(&self) -> BigUint {
        F::Params::MODULUS.into()
    }

    /// Add an r1cs constraint that is 3 linear combinations.
    /// This represents one constraint: a * b = c
    fn add_constraint(&mut self, note: &str, c: Constraint<F>, span: Span) {
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
        self.witness_vector.len() - self.public_inputs.len() - self.public_outputs.len()
    }

    fn enforce_constraint(
        &mut self,
        a: &LinearCombination<F>,
        b: &LinearCombination<F>,
        c: &LinearCombination<F>,
        span: Span,
    ) {
        self.add_constraint(
            "enforce constraint",
            Constraint {
                a: a.clone(),
                b: b.clone(),
                c: c.clone(),
            },
            span,
        );
    }
}

#[derive(Debug)]
/// An intermediate struct for `SnarkjsExporter` to reorder the witness and convert to snarkjs format.
pub struct GeneratedWitness<F>
where
    F: BackendField,
{
    pub witness: Vec<F>,
    pub outputs: Vec<F>,
}

impl<F> Backend for R1CS<F>
where
    F: BackendField,
{
    type Field = F;
    type Var = LinearCombination<F>;
    type GeneratedWitness = GeneratedWitness<F>;

    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon::<F>
    }

    fn init_circuit(&mut self) {
        // create the first var that is always 1
        self.new_internal_var(Value::Constant(F::one()), Span::default());
    }

    /// Create a new `CellVar` and record in `witness_vector` vector.
    /// The underlying type of `CellVar` is always `WitnessVar`.
    fn new_internal_var(
        &mut self,
        val: crate::var::Value<Self>,
        span: Span,
    ) -> LinearCombination<F> {
        let var = CellVar {
            index: self.witness_vector.len(),
            span,
        };

        self.witness_vector.insert(var.index, val);

        LinearCombination::from(var)
    }

    fn add_constant(
        &mut self,
        //todo: do we need this?
        label: Option<&'static str>,
        value: F,
        span: Span,
    ) -> LinearCombination<F> {
        let x = self.new_internal_var(Value::Constant(value), span);
        self.assert_eq_const(&x, value, span);

        x
    }

    /// Final checks for generating the circuit.
    /// todo: we might need to rethink about this interface
    /// - we might just need the `returned_cells` argument, as the backend can record the public outputs itself?
    fn finalize_circuit(
        &mut self,
        public_output: Option<crate::var::Var<Self::Field, Self::Var>>,
        returned_cells: Option<Vec<LinearCombination<F>>>,
        main_span: Span,
    ) -> crate::error::Result<()> {
        // store the return value in the public input that was created for that
        if let Some(public_output) = public_output {
            let cvars = &public_output.cvars;

            for (pub_var, ret_var) in cvars.clone().iter().zip(returned_cells.unwrap()) {
                // replace the computation of the public output vars with the actual variables being returned here
                let var_idx = pub_var.cvar().unwrap().to_cell_var().index;
                let prev = &self.witness_vector[var_idx];
                assert!(matches!(prev, Value::PublicOutput(None)));
                self.witness_vector[var_idx] = Value::PublicOutput(Some(ret_var));
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
        for (index, _) in self.witness_vector.iter().enumerate() {
            // Skip the first var which is always 1
            // - In a linear combination, each of the vars can be paired with a coefficient.
            // - The first var is assumed to be the factor of the constant of a linear combination.
            if index == 0 {
                continue;
            }

            if !written_vars.contains(&index) {
                if let Some(private_cell_var) = self
                    .private_input_cell_vars
                    .iter()
                    .find(|private_cell_var| private_cell_var.index == index)
                {
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

        self.finalized = true;

        Ok(())
    }

    fn compute_var(
        &self,
        env: &mut crate::witness::WitnessEnv<Self::Field>,
        lc: &LinearCombination<Self::Field>,
    ) -> Result<Self::Field> {
        let mut val = lc.constant;

        for (var, factor) in &lc.terms {
            let var_val = self.witness_vector.get(var.index).unwrap();
            let calc = self.compute_val(env, var_val, var.index)? * factor;
            val += calc;
        }

        Ok(val)
    }

    /// Generate the witnesses
    /// This process should check if the constraints are satisfied.
    fn generate_witness(
        &self,
        witness_env: &mut crate::witness::WitnessEnv<F>,
    ) -> crate::error::Result<Self::GeneratedWitness> {
        assert!(self.finalized, "the circuit is not finalized yet!");

        // generate witness through witness vars vector
        let mut witness = self
            .witness_vector
            .iter()
            .enumerate()
            .map(|(index, val)| {
                match val {
                    // Defer calculation for output vars.
                    // The reasoning behind this is to avoid deep recursion potentially triggered by the public output var at the beginning.
                    Value::PublicOutput(_) => Ok(F::zero()),
                    _ => self.compute_val(witness_env, val, index),
                }
            })
            .collect::<crate::error::Result<Vec<F>>>()?;

        // The original vars of public outputs are not part of the constraints
        // so we need to compute them separately
        for var in &self.public_outputs {
            let val = self.compute_var(witness_env, &var.to_linear_combination())?;
            witness[var.index] = val;
        }

        for (index, (constraint, debug_info)) in
            izip!(&self.constraints, &self.debug_info).enumerate()
        {
            // assert a * b = c
            let ab = constraint.a.evaluate(&witness) * constraint.b.evaluate(&witness);
            let c = constraint.c.evaluate(&witness);

            if ab != c {
                return Err(Error::new(
                    "runtime",
                    ErrorKind::InvalidWitness(index),
                    debug_info.span,
                ));
            }
        }

        let outputs = self
            .public_outputs
            .iter()
            .map(|var| witness[var.index])
            .collect();

        Ok(GeneratedWitness { witness, outputs })
    }

    fn generate_asm(&self, sources: &crate::compiler::Sources, debug: bool) -> String {
        let zero = F::zero();

        let mut res = String::new();
        res.push_str(&crate::utils::noname_version());

        for ((row, constraint), debug_info) in
            izip!(self.constraints.iter().enumerate(), &self.debug_info)
        {
            if debug {
                // first info row: show the current row of constraints
                res.push_str(&format!("╭{}\n", "─".repeat(80)));
                res.push_str(&format!("│ {row} │ "));
            }

            // format the a b c linear combinations in order
            let fmt_lcs: Vec<String> = constraint
                .as_array()
                .iter()
                .map(|lc| {
                    let mut terms: Vec<String> = lc
                        .terms
                        .iter()
                        // sort by var index to make it determisitic for asm generation
                        .sorted_by(|(a, _), (b, _)| a.index.cmp(&b.index))
                        .map(|(var, factor)| {
                            match factor.pretty().as_str() {
                                // if the factor is 1, we don't need to show it
                                "1" => format!("v_{}", var.index),
                                _ => format!("{} * v_{}", factor.pretty(), var.index),
                            }
                        })
                        .collect();

                    // ignore the constant if it's zero
                    if lc.constant != zero {
                        terms.push(lc.constant.pretty());
                    }

                    // check if it needs to cancatenate the terms with a plus sign
                    match terms.len() {
                        0 => "0".to_string(),
                        1 => terms[0].clone(),
                        _ => terms.join(" + "),
                    }
                })
                .collect();

            let (a, b, c) = (&fmt_lcs[0], &fmt_lcs[1], &fmt_lcs[2]);

            // format an entire constraint
            res.push_str(&format!("{c} == ({a}) * ({b})\n"));

            if debug {
                // link the constraint to the source code
                crate::utils::display_source(&mut res, sources, &[debug_info.clone()]);
            }
        }

        res
    }

    fn neg(&mut self, x: &LinearCombination<F>, span: Span) -> LinearCombination<F> {
        let one = F::one();
        let x = x.clone();

        x.scale(one.neg(), span)
    }

    fn add(
        &mut self,
        lhs: &LinearCombination<F>,
        rhs: &LinearCombination<F>,
        span: Span,
    ) -> LinearCombination<F> {
        lhs.add(rhs, span)
    }

    fn add_const(&mut self, x: &LinearCombination<F>, cst: &F, span: Span) -> LinearCombination<F> {
        let cst_lc = LinearCombination::from_const(*cst, span);
        x.add(&cst_lc, span)
    }

    fn mul(
        &mut self,
        lhs: &LinearCombination<F>,
        rhs: &LinearCombination<F>,
        span: Span,
    ) -> LinearCombination<F> {
        lhs.mul(self, rhs, span)
    }

    fn mul_const(&mut self, x: &LinearCombination<F>, cst: &F, span: Span) -> LinearCombination<F> {
        x.scale(*cst, span)
    }

    fn assert_eq_const(&mut self, x: &LinearCombination<F>, cst: F, span: Span) {
        let c = LinearCombination::from_const(cst, span);

        x.assert_eq(self, &c, span);
    }

    fn assert_eq_var(
        &mut self,
        lhs: &LinearCombination<F>,
        rhs: &LinearCombination<F>,
        span: Span,
    ) {
        lhs.assert_eq(self, rhs, span);
    }

    /// Adds the public input cell vars.
    fn add_public_input(&mut self, val: Value<Self>, span: Span) -> LinearCombination<F> {
        let var = self.new_internal_var(val, span);
        self.public_inputs.push(*var.to_cell_var());

        var
    }

    /// Adds the private input cell vars.
    fn add_private_input(&mut self, val: Value<Self>, span: Span) -> LinearCombination<F> {
        let var = self.new_internal_var(val, span);
        self.private_input_cell_vars.push(*var.to_cell_var());

        var
    }

    /// Adds the public output cell vars.
    fn add_public_output(&mut self, val: Value<Self>, span: Span) -> LinearCombination<F> {
        let var = self.new_internal_var(val, span);
        self.public_outputs.push(*var.to_cell_var());

        var
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::{
        r1cs::{R1csBls12381Field, R1CS},
        Backend, BackendKind,
    };
    use ark_ff::One;
    use rstest::rstest;

    #[rstest]
    #[case::bls12381(BackendKind::new_r1cs_bls12_381())]
    #[case::bn254(BackendKind::new_r1cs_bn254())]
    fn test_prime(#[case] r1cs: BackendKind) {
        match r1cs {
            BackendKind::R1csBls12_381(r1cs) => {
                let prime = r1cs.prime().to_string();
                assert_eq!(
                    prime,
                    "52435875175126190479447740508185965837690552500527637822603658699938581184513"
                );
            }
            BackendKind::R1csBn254(r1cs) => {
                let prime = r1cs.prime().to_string();
                assert_eq!(
                    prime,
                    "21888242871839275222246405745257275088548364400416034343698204186575808495617"
                );
            }
            _ => {
                panic!("unexpected backend kind")
            }
        }
    }

    #[test]
    fn test_init_circuit() {
        let mut r1cs: R1CS<R1csBls12381Field> = R1CS::new();
        r1cs.init_circuit();

        // first var should be initialized as 1
        assert_eq!(r1cs.witness_vector.len(), 1);
        match &r1cs.witness_vector[0] {
            crate::var::Value::Constant(cst) => {
                assert_eq!(*cst, R1csBls12381Field::one());
            }
            _ => panic!("unexpected value"),
        }
    }
}

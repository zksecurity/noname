pub mod builtin;
pub mod snarkjs;

use std::collections::{HashMap, HashSet};

use ark_ff::FpParameters;
use itertools::{izip, Itertools as _};
use num_bigint::BigUint;

use crate::error::{Error, ErrorKind};
use crate::helpers::PrettyField;
use crate::{
    circuit_writer::DebugInfo,
    var::{CellVar, Value},
};

use super::{Backend, BackendField};

pub type R1csBls12381Field = ark_bls12_381::Fr;
pub type R1csBn254Field = ark_bn254::Fr;

// Because the associated field type is BackendField, we need to implement it for the actual field types in order to use them.
impl BackendField for R1csBls12381Field {}
impl BackendField for R1csBn254Field {}

/// Linear combination of variables and constants.
/// For example, the linear combination is represented as a * f_a + b * f_b + f_c
/// f_a and f_b are the coefficients of a and b respectively.
/// a and b are represented by CellVar.
/// The constant f_c is represented by the constant field, will always multiply with the variable at index 0 which is always 1.
#[derive(Clone, Debug)]
pub struct LinearCombination<F>
where
    F: BackendField,
{
    pub terms: HashMap<CellVar, F>,
    pub constant: F,
}

impl<F> LinearCombination<F>
where
    F: BackendField,
{
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
    fn one() -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: F::one(),
        }
    }

    /// Create a linear combination to represent constant zero.
    fn zero() -> Self {
        LinearCombination {
            terms: HashMap::new(),
            constant: F::zero(),
        }
    }

    /// Create a linear combination from a list of vars
    fn from_vars(vars: Vec<CellVar>) -> Self {
        let terms = vars.into_iter().map(|var| (var, F::one())).collect();
        LinearCombination {
            terms,
            constant: F::zero(),
        }
    }

    /// Create a linear combination from a constant.
    fn from_const(cst: F) -> Self {
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

/// R1CS backend with bls12_381 field.
#[derive(Clone)]
pub struct R1CS<F>
where
    F: BackendField,
{
    /// Constraints in the r1cs.
    constraints: Vec<Constraint<F>>,
    witness_vars: Vec<Value<Self>>,
    debug_info: Vec<DebugInfo>,
    /// Record the public inputs for reordering the witness vector
    public_inputs: Vec<CellVar>,
    /// Record the public outputs for reordering the witness vector
    public_outputs: Vec<CellVar>,
    finalized: bool,
}

impl<F> R1CS<F>
where
    F: BackendField,
{
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

    /// Returns the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the prime for snarkjs based on the curve field.
    fn prime(&self) -> BigUint {
        F::Params::MODULUS.into()
    }

    /// Add an r1cs constraint that is 3 linear combinations.
    /// This represents one constraint: a * b = c
    fn add_constraint(&mut self, note: &str, c: Constraint<F>, span: crate::constants::Span) {
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

    type GeneratedWitness = GeneratedWitness<F>;

    fn witness_vars(&self, var: CellVar) -> &Value<Self> {
        self.witness_vars.get(var.index).unwrap()
    }

    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon::<F>
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
        value: F,
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
        public_output: Option<crate::var::Var<F>>,
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
        for (index, _) in self.witness_vars.iter().enumerate() {
            assert!(
                written_vars.contains(&index),
                "there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!"
            );
        }

        self.finalized = true;

        Ok(())
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
            .witness_vars
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
            let val = self.compute_var(witness_env, *var)?;
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
                            // starting from index 1, as the first var is reserved for the constant
                            let index = var.index + 1;
                            match factor.pretty().as_str() {
                                // if the factor is 1, we don't need to show it
                                "1" => format!("v_{}", index),
                                _ => format!("{} * v_{}", factor.pretty(), index),
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
            res.push_str(&format!("{} == ({}) * ({})\n", c, a, b));

            if debug {
                // link the constraint to the source code
                crate::utils::display_source(&mut res, sources, &[debug_info.clone()]);
            }
        }

        res
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
        let one = F::one();
        let zero = F::zero();

        let x_neg =
            self.new_internal_var(Value::LinearCombination(vec![(one.neg(), *x)], zero), span);

        let a = LinearCombination::from_vars(vec![*x, x_neg]);
        let b = LinearCombination::one();
        let c = LinearCombination::zero();

        self.add_constraint("neg constraint: x + (-x) = 0", Constraint { a, b, c }, span);

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
        let one = F::one();
        let zero = F::zero();

        let res = self.new_internal_var(
            Value::LinearCombination(vec![(one, *lhs), (one, *rhs)], zero),
            span,
        );

        // IMPORTANT: since terms use CellVar as key,
        // HashMap automatically overrides it to single one term if the two vars are the same CellVar
        let a = if lhs == rhs {
            LinearCombination {
                terms: HashMap::from_iter(vec![(*lhs, F::from(2u32))]),
                constant: zero,
            }
        } else {
            LinearCombination::from_vars(vec![*lhs, *rhs])
        };

        let b = LinearCombination::one();
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "add constraint: lhs + rhs = res",
            Constraint { a, b, c },
            span,
        );

        res
    }

    fn add_const(&mut self, x: &CellVar, cst: &F, span: crate::constants::Span) -> CellVar {
        // To constrain:
        // x + cst = res
        // given:
        // a * b = c
        // then:
        // a = x + cst
        // b = 1
        // c = res
        let one = F::one();

        let res = self.new_internal_var(Value::LinearCombination(vec![(one, *x)], *cst), span);

        let a = LinearCombination {
            terms: HashMap::from_iter(vec![(*x, one)]),
            constant: *cst,
        };

        let b = LinearCombination::one();
        let c = LinearCombination::from_vars(vec![res]);

        self.add_constraint(
            "add constraint: x + cst = res",
            Constraint { a, b, c },
            span,
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
            Constraint { a, b, c },
            span,
        );

        res
    }

    fn mul_const(&mut self, x: &CellVar, cst: &F, span: crate::constants::Span) -> CellVar {
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
            Constraint { a, b, c },
            span,
        );

        res
    }

    fn assert_eq_const(&mut self, x: &CellVar, cst: F, span: crate::constants::Span) {
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

        self.add_constraint("eq constraint: x = cst", Constraint { a, b, c }, span);
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

        self.add_constraint("eq constraint: lhs = rhs", Constraint { a, b, c }, span);
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

mod tests {
    use crate::backends::BackendKind;
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
}

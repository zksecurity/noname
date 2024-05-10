use std::{collections::HashMap, str::FromStr};

use ark_ff::{Field, Zero};
use num_bigint::BigUint;

use crate::{
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    imports::FnHandle,
    var::{CellVar, Value, Var},
    witness::WitnessEnv,
};

use self::{kimchi::KimchiVesta, r1cs::R1CS};

pub mod kimchi;
pub mod r1cs;

/// This trait serves as an alias for a bundle of traits
pub trait BackendField:
    Field + FromStr + TryFrom<BigUint> + TryInto<BigUint> + Into<BigUint> + PrettyField
{
}

pub enum BackendKind {
    KimchiVesta(KimchiVesta),
    R1csBls12_381(R1CS<ark_bls12_381::Fr>),
    R1csBn128(R1CS<ark_bn254::Fr>),
}

impl BackendKind {
    pub fn new_kimchi_vesta(use_double_generic: bool) -> Self {
        Self::KimchiVesta(KimchiVesta::new(use_double_generic))
    }

    pub fn new_r1cs_bls12_381() -> Self {
        Self::R1csBls12_381(R1CS::new())
    }

    pub fn new_r1cs_bn128() -> Self {
        Self::R1csBn128(R1CS::new())
    }
}

// TODO: should it be cloneable? It is now so because FnInfo needs to be cloneable.
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: BackendField;

    /// The generated witness type for the backend. Each backend may define its own witness format to be generated.
    type GeneratedWitness;

    /// This provides a standard way to access to all the internal vars.
    /// Different backends should be accessible in the same way by the variable index.
    fn witness_vars(&self, var: CellVar) -> &Value<Self>;

    // TODO: as the builtins grows, we might better change this to a crypto struct that holds all the builtin function pointers.
    /// poseidon crypto builtin function for different backends
    fn poseidon() -> FnHandle<Self>;

    /// Create a new cell variable and record it.
    /// It increments the variable index for look up later.
    fn new_internal_var(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// negate a var
    fn neg(&mut self, var: &CellVar, span: Span) -> CellVar;

    /// add two vars
    fn add(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span) -> CellVar;

    /// add a var with a constant
    fn add_const(&mut self, var: &CellVar, cst: &Self::Field, span: Span) -> CellVar;

    /// multiply a var with another var
    fn mul(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span) -> CellVar;

    /// multiply a var with a constant
    fn mul_const(&mut self, var: &CellVar, cst: &Self::Field, span: Span) -> CellVar;

    /// add a constraint to assert a var equals a constant
    fn assert_eq_const(&mut self, var: &CellVar, cst: Self::Field, span: Span);

    /// add a constraint to assert a var equals another var
    fn assert_eq_var(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span);

    /// Process a public input
    fn add_public_input(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// Process a public output
    fn add_public_output(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Self::Field,
        span: Span,
    ) -> CellVar;

    /// Compute the value of the symbolic cell variables.
    /// It recursively does the computation down the stream until it is not a symbolic variable.
    /// - The symbolic variables are stored in the witness_vars.
    /// - The computed values are stored in the cached_values.
    fn compute_var(&self, env: &mut WitnessEnv<Self::Field>, var: CellVar) -> Result<Self::Field> {
        self.compute_val(env, self.witness_vars(var), var.index)
    }

    fn compute_val(
        &self,
        env: &mut WitnessEnv<Self::Field>,
        val: &Value<Self>,
        var_index: usize,
    ) -> Result<Self::Field> {
        if let Some(res) = env.cached_values.get(&var_index) {
            return Ok(*res);
        }

        match &val {
            Value::Hint(func) => {
                let res = func(self, env)
                    .expect("that function doesn't return a var (type checker error)");
                env.cached_values.insert(var_index, res);
                Ok(res)
            }
            Value::Constant(c) => Ok(*c),
            Value::LinearCombination(lc, cst) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_var(env, *var)? * *coeff;
                }
                env.cached_values.insert(var_index, res); // cache
                Ok(res)
            }
            Value::Mul(lhs, rhs) => {
                let lhs = self.compute_var(env, *lhs)?;
                let rhs = self.compute_var(env, *rhs)?;
                let res = lhs * rhs;
                env.cached_values.insert(var_index, res); // cache
                Ok(res)
            }
            Value::Inverse(v) => {
                let v = self.compute_var(env, *v)?;
                let res = v.inverse().unwrap_or_else(Self::Field::zero);
                env.cached_values.insert(var_index, res); // cache
                Ok(res)
            }
            Value::External(name, idx) => Ok(env.get_external(name)[*idx]),
            Value::PublicOutput(var) => {
                // var can be none. what could be the better way to pass in the span in that case?
                // let span = self.main_info().span;
                let var = var.ok_or_else(|| {
                    Error::new("runtime", ErrorKind::MissingReturn, Span::default())
                })?;
                self.compute_var(env, var)
            }
            Value::Scale(scalar, var) => {
                let var = self.compute_var(env, *var)?;
                Ok(*scalar * var)
            }
        }
    }

    /// Finalize the circuit by doing some sanitizing checks.
    fn finalize_circuit(
        &mut self,
        public_output: Option<Var<Self::Field>>,
        returned_cells: Option<Vec<CellVar>>,
        private_input_indices: Vec<usize>,
        main_span: Span,
    ) -> Result<()>;

    /// Generate the witness for a backend.
    fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<Self::Field>,
    ) -> Result<Self::GeneratedWitness>;

    /// Generate the asm for a backend.
    fn generate_asm(&self, sources: &Sources, debug: bool) -> String;
}

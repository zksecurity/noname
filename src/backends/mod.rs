use std::{fmt::Debug, str::FromStr};

use ark_ff::{Field, Zero};
use num_bigint::BigUint;

use crate::{
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    imports::FnHandle,
    var::{Value, Var},
    witness::WitnessEnv,
};

use self::{
    kimchi::KimchiVesta,
    r1cs::{R1csBls12381Field, R1csBn254Field, R1CS},
};

pub mod kimchi;
pub mod r1cs;

/// This trait serves as an alias for a bundle of traits
pub trait BackendField:
    Field + FromStr + TryFrom<BigUint> + TryInto<BigUint> + Into<BigUint> + PrettyField
{
}

/// This trait allows different backends to have different cell var types.
/// It is intended to make it opaque to the frondend.
pub trait BackendVar: Clone + Debug + PartialEq + Eq {}

// TODO: Fix this by `Box`ing `KimchiVesta`.
#[allow(clippy::large_enum_variant)]
pub enum BackendKind {
    KimchiVesta(KimchiVesta),
    R1csBls12_381(R1CS<R1csBls12381Field>),
    R1csBn254(R1CS<R1csBn254Field>),
}

impl BackendKind {
    #[must_use]
    pub fn new_kimchi_vesta(use_double_generic: bool) -> Self {
        Self::KimchiVesta(KimchiVesta::new(use_double_generic))
    }

    #[must_use]
    pub fn new_r1cs_bls12_381() -> Self {
        Self::R1csBls12_381(R1CS::new())
    }

    #[must_use]
    pub fn new_r1cs_bn254() -> Self {
        Self::R1csBn254(R1CS::new())
    }
}

// TODO: should it be cloneable? It is now so because FnInfo needs to be cloneable.
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: BackendField;

    /// The `CellVar` type for the backend.
    /// Different backend is allowed to have different `CellVar` types.
    type Var: BackendVar;

    /// The generated witness type for the backend. Each backend may define its own witness format to be generated.
    type GeneratedWitness;

    // TODO: as the builtins grows, we might better change this to a crypto struct that holds all the builtin function pointers.
    /// poseidon crypto builtin function for different backends
    fn poseidon() -> FnHandle<Self>;

    /// Init circuit
    fn init_circuit(&mut self) {
        // do nothing by default
    }

    /// Create a new cell variable and record it.
    /// It increments the variable index for look up later.
    fn new_internal_var(&mut self, val: Value<Self>, span: Span) -> Self::Var;

    /// negate a var
    fn neg(&mut self, var: &Self::Var, span: Span) -> Self::Var;

    /// add two vars
    fn add(&mut self, lhs: &Self::Var, rhs: &Self::Var, span: Span) -> Self::Var;

    /// sub two vars
    fn sub(&mut self, lhs: &Self::Var, rhs: &Self::Var, span: Span) -> Self::Var {
        let rhs_neg = self.neg(rhs, span);
        self.add(lhs, &rhs_neg, span)
    }

    /// add a var with a constant
    fn add_const(&mut self, var: &Self::Var, cst: &Self::Field, span: Span) -> Self::Var;

    /// multiply a var with another var
    fn mul(&mut self, lhs: &Self::Var, rhs: &Self::Var, span: Span) -> Self::Var;

    /// multiply a var with a constant
    fn mul_const(&mut self, var: &Self::Var, cst: &Self::Field, span: Span) -> Self::Var;

    /// add a constraint to assert a var equals a constant
    fn assert_eq_const(&mut self, var: &Self::Var, cst: Self::Field, span: Span);

    /// add a constraint to assert a var equals another var
    fn assert_eq_var(&mut self, lhs: &Self::Var, rhs: &Self::Var, span: Span);

    /// Process a public input
    fn add_public_input(&mut self, val: Value<Self>, span: Span) -> Self::Var;

    /// Process a private input
    fn add_private_input(&mut self, val: Value<Self>, span: Span) -> Self::Var;

    /// Process a public output
    fn add_public_output(&mut self, val: Value<Self>, span: Span) -> Self::Var;

    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Self::Field,
        span: Span,
    ) -> Self::Var;

    /// Backends should implement this function to load and compute the value of a `CellVar`.
    fn compute_var(
        &self,
        env: &mut WitnessEnv<Self::Field>,
        var: &Self::Var,
    ) -> Result<Self::Field>;

    fn compute_val(
        &self,
        env: &mut WitnessEnv<Self::Field>,
        val: &Value<Self>,
        cache_key: usize,
    ) -> Result<Self::Field> {
        if let Some(res) = env.cached_values.get(&cache_key) {
            return Ok(*res);
        }

        match val {
            Value::Hint(func) => {
                let res = func(self, env)
                    .expect("that function doesn't return a var (type checker error)");
                env.cached_values.insert(cache_key, res);
                Ok(res)
            }
            Value::Constant(c) => Ok(*c),
            Value::LinearCombination(lc, cst) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_var(env, var)? * *coeff;
                }
                env.cached_values.insert(cache_key, res); // cache
                Ok(res)
            }
            Value::Mul(lhs, rhs) => {
                let lhs = self.compute_var(env, lhs)?;
                let rhs = self.compute_var(env, rhs)?;
                let res = lhs * rhs;
                env.cached_values.insert(cache_key, res); // cache
                Ok(res)
            }
            Value::Inverse(v) => {
                let v = self.compute_var(env, v)?;
                let res = v.inverse().unwrap_or_else(Self::Field::zero);
                env.cached_values.insert(cache_key, res); // cache
                Ok(res)
            }
            Value::External(name, idx) => Ok(env.get_external(name)[*idx]),
            Value::PublicOutput(var) => {
                // var can be none. what could be the better way to pass in the span in that case?
                // let span = self.main_info().span;
                let var = var.as_ref().ok_or_else(|| {
                    Error::new("runtime", ErrorKind::MissingReturn, Span::default())
                })?;
                self.compute_var(env, var)
            }
            Value::Scale(scalar, var) => {
                let var = self.compute_var(env, var)?;
                Ok(*scalar * var)
            }
        }
    }

    /// Finalize the circuit by doing some sanitizing checks.
    fn finalize_circuit(
        &mut self,
        public_output: Option<Var<Self::Field, Self::Var>>,
        returned_cells: Option<Vec<Self::Var>>,
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

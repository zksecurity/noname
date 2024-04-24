use std::collections::HashMap;

use ark_ff::{Field, Zero};

use crate::{
    circuit_writer::{DebugInfo, GateKind},
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    imports::FnHandle,
    var::{CellVar, ConstOrCell, Value, Var},
    witness::WitnessEnv,
};

pub mod kimchi;

// TODO: should it be cloneable? It is now so because FnInfo needs to be cloneable.
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: Field + PrettyField;

    /// The generated witness type for the backend. Each backend may define its own witness format to be generated.
    type GeneratedWitness;

    /// This provides a standard way to access to all the internal vars.
    /// Different backends should be accessible in the same way by the variable index.
    fn witness_vars(&self) -> &HashMap<usize, Value<Self>>;

    // TODO: as the builtins grows, we might better change this to a crypto struct that holds all the builtin function pointers.
    /// poseidon crypto builtin function for different backends
    fn poseidon() -> FnHandle<Self>;

    /// Create a new cell variable and record it.
    /// It increments the variable index for look up later.
    fn new_internal_var(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// basic constraint negation
    fn constraint_neg(&mut self, var: &CellVar, span: Span) -> CellVar;

    /// add a constraint to assert two vars are added together
    fn constraint_add(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span) -> CellVar;

    /// add a constraint to assert a var is added to a constant
    fn constraint_add_const(&mut self, var: &CellVar, cst: &Self::Field, span: Span) -> CellVar;

    /// add a constraint to assert a var is multiplied by another var
    fn constraint_mul(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span) -> CellVar;

    /// add a constraint to assert a var is multiplied by a constant
    fn constraint_mul_const(&mut self, var: &CellVar, cst: &Self::Field, span: Span) -> CellVar;

    /// add a constraint to assert a var equals a constant
    fn constraint_eq_const(&mut self, var: &CellVar, cst: Self::Field, span: Span);

    /// add a constraint to assert a var equals another var
    fn constraint_eq_var(&mut self, lhs: &CellVar, rhs: &CellVar, span: Span);

    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Self::Field,
        span: Span,
    ) -> CellVar;

    /// Add a constraint for a public input
    fn constraint_public_input(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// Add a constraint for a public output
    fn constraint_public_output(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// Compute the value of the symbolic cell variables.
    /// It recursively does the computation down the stream until it is not a symbolic variable.
    /// - The symbolic variables are stored in the witness_vars.
    /// - The computed values are stored in the cached_values.
    fn compute_var(&self, env: &mut WitnessEnv<Self::Field>, var: CellVar) -> Result<Self::Field> {
        // fetch cache first
        // TODO: if self was &mut, then we could use a Value::Cached(Field) to store things instead of that
        if let Some(res) = env.cached_values.get(&var) {
            return Ok(*res);
        }

        match &self.witness_vars()[&var.index] {
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
                let res = v.inverse().unwrap_or_else(Self::Field::zero);
                env.cached_values.insert(var, res); // cache
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

    // TODO: we may need to move the finalized flag from circuit writer to backend, so the backend can freeze itself once finalized.
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

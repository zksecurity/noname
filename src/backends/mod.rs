use std::collections::HashMap;

use ark_ff::{Field, Zero};

use crate::{
    circuit_writer::{DebugInfo, GateKind}, compiler::{GeneratedWitness, Sources}, constants::Span, error::{Error, ErrorKind, Result}, helpers::PrettyField, var::{CellVar, Value, Var}, witness::WitnessEnv
};

pub mod kimchi;
pub mod r1cs;

/// Each backend implements this trait
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: Field + PrettyField;

    /// This provides a standard way to access to all the internal vars.
    /// Different backends should be accessible in the same way by the variable index.
    fn witness_vars(&self) -> &HashMap<usize, Value<Self>>;

    fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    fn add_generic_gate(
        &mut self,
        label: &'static str,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Self::Field>,
        span: Span,
    );

    fn debug_info(&self) -> &[DebugInfo];

    fn new_internal_var(&mut self, val: Value<Self>, span: Span) -> CellVar;

    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Self::Field,
        span: Span,
    ) -> CellVar;
    
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
                let var =
                    var.ok_or_else(|| Error::new("runtime", ErrorKind::MissingReturn, Span::default()))?;
                self.compute_var(env, var)
            }
            Value::Scale(scalar, var) => {
                let var = self.compute_var(env, *var)?;
                Ok(*scalar * var)
            }
        }
    }

    fn finalize_circuit(
        &mut self, 
        public_output: Option<Var<Self::Field>>, 
        returned_cells: Option<Vec<CellVar>>,
        private_input_indices: Vec<usize>, 
        main_span: Span
    ) -> Result<()>;

    fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<Self::Field>,
        public_input_size: usize,
    ) -> Result<GeneratedWitness<Self>>;

    fn generate_asm(&self, sources: &Sources, debug: bool) -> String;
}

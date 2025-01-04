use std::{fmt::Debug, str::FromStr};

use ::kimchi::o1_utils::FieldHelpers;
use ark_ff::{Field, One, PrimeField, Zero};
use circ::ir::term::precomp::PreComp;
use fxhash::FxHashMap;
use num_bigint::BigUint;

use crate::{
    circuit_writer::{ir::IRWriter, VarInfo},
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    imports::FnHandle,
    var::{ConstOrCell, Value, Var},
    witness::WitnessEnv,
};

use self::{
    kimchi::KimchiVesta,
    r1cs::{R1csBls12381Field, R1csBn254Field, R1CS},
};

use crate::mast::Mast;

pub mod kimchi;
pub mod r1cs;

/// This trait serves as an alias for a bundle of traits
pub trait BackendField:
    Field + FromStr + TryFrom<BigUint> + TryInto<BigUint> + Into<BigUint> + PrettyField
{
    fn to_circ_field(&self) -> circ_fields::FieldV;
    fn to_circ_type() -> circ_fields::FieldT;
}

/// This trait allows different backends to have different cell var types.
/// It is intended to make it opaque to the frondend.
pub trait BackendVar: Clone + Debug + PartialEq + Eq {}

pub enum BackendKind {
    KimchiVesta(KimchiVesta),
    R1csBls12_381(R1CS<R1csBls12381Field>),
    R1csBn254(R1CS<R1csBn254Field>),
}

impl BackendKind {
    pub fn new_kimchi_vesta(use_double_generic: bool) -> Self {
        Self::KimchiVesta(KimchiVesta::new(use_double_generic))
    }

    pub fn new_r1cs_bls12_381() -> Self {
        Self::R1csBls12_381(R1CS::new())
    }

    pub fn new_r1cs_bn254() -> Self {
        Self::R1csBn254(R1CS::new())
    }
}

// TODO: should it be cloneable? It is now so because FnInfo needs to be cloneable.
pub trait Backend: Clone {
    /// The circuit field / scalar field that the circuit is written on.
    type Field: BackendField;

    /// The CellVar type for the backend.
    /// Different backend is allowed to have different CellVar types.
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

    /// Backends should implement this function to load and compute the value of a CellVar.
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
            Value::NthBit(var, shift) => {
                let var = self.compute_var(env, var)?;
                let bits = var.to_bits();

                // extract the bit
                let rbit = bits[*shift];

                // convert the bit back to a field element
                let res = if rbit {
                    Self::Field::one()
                } else {
                    Self::Field::zero()
                };

                Ok(res)
            }
            Value::HintIR(t, named_vars, logs) => {
                // map the named vars to env
                let ir_env = named_vars
                    .iter()
                    .map(|(name, var)| {
                        let val = match var {
                            crate::var::ConstOrCell::Const(cst) => cst.to_circ_field(),
                            crate::var::ConstOrCell::Cell(var) => {
                                let val = self.compute_var(env, var).unwrap();
                                val.to_circ_field()
                            }
                        };
                        (name.clone(), circ::ir::term::Value::Field(val))
                    })
                    .collect::<FxHashMap<String, circ::ir::term::Value>>();

                // evaluate logs
                for log in logs {
                    // check the cache on env and log, and only evaluate if not in cache
                    let res: Vec<Self::Field> = IRWriter::<Self>::eval_ir(&ir_env, log);
                    // format and print out array
                    println!(
                        "log: {:#?}",
                        res.iter().map(|f| f.pretty()).collect::<Vec<String>>()
                    );
                }

                // evaluate the term
                let res = IRWriter::<Self>::eval_ir(&ir_env, t)[0];

                env.cached_values.insert(cache_key, res); // cache

                Ok(res)
            }
            Value::Div(lhs, rhs) => {
                let res = match (lhs, rhs) {
                    (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs / rhs;
                        Self::Field::from(res)
                    }
                    (ConstOrCell::Cell(lhs), ConstOrCell::Const(rhs)) => {
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        let lhs = self.compute_var(env, lhs)?;

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs / rhs;

                        Self::Field::from(res)
                    }
                    (ConstOrCell::Const(lhs), ConstOrCell::Cell(rhs)) => {
                        let rhs = self.compute_var(env, rhs)?;
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs / rhs;

                        Self::Field::from(res)
                    }
                    (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
                        let lhs = self.compute_var(env, lhs)?;
                        let rhs = self.compute_var(env, rhs)?;

                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }
                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs / rhs;

                        Self::Field::from(res)
                    }
                };

                env.cached_values.insert(cache_key, res); // cache
                Ok(res)
            }
            Value::Mod(lhs, rhs) => {
                match (lhs, rhs) {
                    (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs % rhs;
                        Ok(Self::Field::from(res))
                    }
                    (ConstOrCell::Cell(lhs), ConstOrCell::Const(rhs)) => {
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        let lhs = self.compute_var(env, lhs)?;

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs % rhs;

                        Ok(Self::Field::from(res))
                    }
                    (ConstOrCell::Const(lhs), ConstOrCell::Cell(rhs)) => {
                        let rhs = self.compute_var(env, rhs)?;
                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }

                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs % rhs;

                        Ok(Self::Field::from(res))
                    }
                    (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
                        let lhs = self.compute_var(env, lhs)?;
                        let rhs = self.compute_var(env, rhs)?;

                        if rhs.is_zero() {
                            return Err(Error::new(
                                "runtime",
                                ErrorKind::DivisionByZero,
                                Span::default(),
                            ));
                        }
                        // convert to bigints
                        let lhs = lhs.to_biguint();
                        let rhs = rhs.to_biguint();

                        let res = lhs % rhs;

                        Ok(Self::Field::from(res))
                    }
                }
            }
        }
    }

    /// Finalize the circuit by doing some sanitizing checks.
    fn finalize_circuit(
        &mut self,
        public_output: Option<Var<Self::Field, Self::Var>>,
        returned_cells: Option<Vec<Self::Var>>,
        disable_safety_check: bool,
    ) -> Result<()>;

    /// Generate the witness for a backend.
    fn generate_witness<B: Backend>(
        &self,
        witness_env: &mut WitnessEnv<Self::Field>,
        sources: &Sources,
        typed: &Mast<B>,
    ) -> Result<Self::GeneratedWitness>;

    /// Generate the asm for a backend.
    fn generate_asm(&self, sources: &Sources, debug: bool) -> String;

    fn log_var(&mut self, var: &VarInfo<Self::Field, Self::Var>, msg: String, span: Span);
}

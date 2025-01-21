use std::{fmt::Debug, str::FromStr};

use ::kimchi::o1_utils::FieldHelpers;
use ark_ff::{Field, One, PrimeField, Zero};
use circ::ir::term::precomp::PreComp;
use fxhash::FxHashMap;
use num_bigint::BigUint;

use crate::{
    circuit_writer::VarInfo,
    compiler::Sources,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    imports::FnHandle,
    parser::types::TyKind,
    utils::{log_array_or_tuple_type, log_custom_type, log_string_type},
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
            Value::HintIR(t, named_vars) => {
                let mut precomp = PreComp::new();
                // For hint evaluation purpose, precomp only has only one output and no connections with other parts,
                // so just use a dummy output var name.
                precomp.add_output("x".to_string(), t.clone());

                // map the named vars to env
                let env = named_vars
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

                // evaluate and get the only one output
                let eval_map = precomp.eval(&env);
                let value = eval_map.get("x").unwrap();
                // convert to field
                let res = match value {
                    circ::ir::term::Value::Field(f) => {
                        let bytes = f.i().to_digits::<u8>(rug::integer::Order::Lsf);
                        Self::Field::from_le_bytes_mod_order(&bytes)
                    }
                    circ::ir::term::Value::Bool(b) => {
                        if *b {
                            Self::Field::one()
                        } else {
                            Self::Field::zero()
                        }
                    }
                    _ => panic!("unexpected output type"),
                };

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
    fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<Self::Field>,
        sources: &Sources,
        typed: &Mast<Self>,
    ) -> Result<Self::GeneratedWitness>;

    /// Generate the asm for a backend.
    fn generate_asm(&self, sources: &Sources, debug: bool) -> String;

    fn log_var(&mut self, var: &VarInfo<Self::Field, Self::Var>, span: Span);

    /// print the log given the log_info
    fn print_log(
        &self,
        witness_env: &mut WitnessEnv<Self::Field>,
        logs: &[(Span, VarInfo<Self::Field, Self::Var>)],
        sources: &Sources,
        typed: &Mast<Self>,
    ) -> Result<()> {
        let mut logs_iter = logs.into_iter();
        while let Some((span, var_info)) = logs_iter.next() {
            let (filename, source) = sources.get(&span.filename_id).unwrap();
            let (line, _, _) = crate::utils::find_exact_line(source, *span);
            let dbg_msg = format!("[{filename}:{line}] -> ");

            match &var_info.typ {
                // Field
                Some(TyKind::Field { .. }) => match &var_info.var[0] {
                    ConstOrCell::Const(cst) => {
                        println!("{dbg_msg}{}", cst.pretty());
                    }
                    ConstOrCell::Cell(cell) => {
                        let val = self.compute_var(witness_env, cell)?;
                        println!("{dbg_msg}{}", val.pretty());
                    }
                },

                // Bool
                Some(TyKind::Bool) => match &var_info.var[0] {
                    ConstOrCell::Const(cst) => {
                        let val = *cst == Self::Field::one();
                        println!("{dbg_msg}{}", val);
                    }
                    ConstOrCell::Cell(cell) => {
                        let val = self.compute_var(witness_env, cell)? == Self::Field::one();
                        println!("{dbg_msg}{}", val);
                    }
                },

                // Array
                Some(TyKind::Array(b, s)) => {
                    let mut typs = Vec::with_capacity(*s as usize);
                    for _ in 0..(*s) {
                        typs.push((**b).clone());
                    }
                    let (output, remaining) = log_array_or_tuple_type(
                        self,
                        &var_info.var.cvars,
                        &typs,
                        *s,
                        witness_env,
                        typed,
                        span,
                        false,
                    )?;
                    assert!(remaining.is_empty());
                    println!("{dbg_msg}{}", output);
                }

                // Custom types
                Some(TyKind::Custom {
                    module,
                    name: struct_name,
                }) => {
                    let mut string_vec = Vec::new();
                    let (output, remaining) = log_custom_type(
                        self,
                        module,
                        struct_name,
                        typed,
                        &var_info.var.cvars,
                        witness_env,
                        span,
                        &mut string_vec,
                    )?;
                    assert!(remaining.is_empty());
                    println!("{dbg_msg}{}{}", struct_name, output);
                }

                // GenericSizedArray
                Some(TyKind::GenericSizedArray(_, _)) => {
                    unreachable!("GenericSizedArray should be monomorphized")
                }

                Some(TyKind::String(s)) => {
                    let output =
                        log_string_type(self, &mut logs_iter, s, witness_env, typed, span)?;
                    println!("{dbg_msg}{}", output);
                }

                Some(TyKind::Tuple(typs)) => {
                    let len = typs.len();
                    let (output, remaining) = log_array_or_tuple_type(
                        self,
                        &var_info.var.cvars,
                        &typs,
                        len as u32,
                        witness_env,
                        typed,
                        span,
                        true,
                    )
                    .unwrap();
                    assert!(remaining.is_empty());
                    println!("{dbg_msg}{}", output);
                }
                None => {
                    return Err(Error::new(
                        "log",
                        ErrorKind::UnexpectedError("No type info for logging"),
                        *span,
                    ))
                }
            }
        }

        Ok(())
    }
}

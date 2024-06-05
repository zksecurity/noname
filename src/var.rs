use std::sync::Arc;

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    backends::{Backend, BackendVar},
    circuit_writer::{CircuitWriter, FnEnv, VarInfo},
    constants::Span,
    error::Result,
    type_checker::ConstInfo,
    witness::WitnessEnv,
};

/// The signature of a hint function
pub type HintFn<B> =
    dyn Fn(&B, &mut WitnessEnv<<B as Backend>::Field>) -> Result<<B as Backend>::Field>;

/// A variable's actual value in the witness can be computed in different ways.
#[derive(Clone, Serialize, Deserialize)]
pub enum Value<B>
where
    B: Backend,
{
    /// Either it's a hint and can be computed from the outside.
    #[serde(skip)]
    // TODO: outch, remove hints? or https://docs.rs/serde_closure/latest/serde_closure/ ?
    // TODO: changed to Arc from Box because Value needs to be cloneable, because of cloneable backend and FnInfo
    // if we can avoid cloning FnInfo, we may be able to avoid this change.
    Hint(Arc<HintFn<B>>),

    /// Or it's a constant (for example, I wrote `2` in the code).
    #[serde(skip)]
    Constant(B::Field),

    /// Or it's a linear combination of internal circuit variables (+ a constant).
    // TODO: probably values of internal variables should be cached somewhere
    #[serde(skip)]
    LinearCombination(Vec<(B::Field, B::Var)>, B::Field /* cst */),

    Mul(B::Var, B::Var),

    #[serde(skip)]
    Scale(B::Field, B::Var),

    /// Returns the inverse of the given variable.
    /// Note that it will potentially return 0 if the given variable is 0.
    Inverse(B::Var),

    /// A public or private input to the function
    /// There's an index associated to a variable name, as the variable could be composed of several field elements.
    External(String, usize),

    /// A public output.
    /// This is tracked separately as public inputs as it needs to be computed later.
    PublicOutput(Option<B::Var>),
}

impl<B: Backend> std::fmt::Debug for Value<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Hint(..) => write!(f, "Hint"),
            Value::Constant(..) => write!(f, "Constant"),
            Value::LinearCombination(..) => write!(f, "LinearCombination"),
            Value::Mul(..) => write!(f, "Mul"),
            Value::Inverse(_) => write!(f, "Inverse"),
            Value::External(..) => write!(f, "External"),
            Value::PublicOutput(..) => write!(f, "PublicOutput"),
            Value::Scale(..) => write!(f, "Scaling"),
        }
    }
}

/// Represents a cell in the execution trace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConstOrCell<F, C>
where
    F: Field,
    C: BackendVar,
{
    /// A constant value.
    #[serde(skip)]
    Const(F),

    /// A cell in the execution trace.
    Cell(C),
}

impl<F: Field, C: BackendVar> ConstOrCell<F, C> {
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    pub fn cst(&self) -> Option<F> {
        match self {
            Self::Const(cst) => Some(*cst),
            _ => None,
        }
    }

    pub fn cvar(&self) -> Option<&C> {
        match self {
            Self::Cell(cvar) => Some(cvar),
            _ => None,
        }
    }
}

/// Represents a variable in the noname language, or an anonymous variable during computation of expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Var<F, C>
where
    F: Field,
    C: BackendVar,
{
    /// The type of variable.
    pub cvars: Vec<ConstOrCell<F, C>>,

    /// The span that created the variable.
    pub span: Span,
}

impl<F: Field, C: BackendVar> Var<F, C> {
    #[must_use]
    pub fn new(cvars: Vec<ConstOrCell<F, C>>, span: Span) -> Self {
        Self { cvars, span }
    }

    pub fn new_cvar(cvar: ConstOrCell<F, C>, span: Span) -> Self {
        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_var(cvar: C, span: Span) -> Self {
        Self {
            cvars: vec![ConstOrCell::Cell(cvar)],
            span,
        }
    }

    pub fn new_constant(cst: F, span: Span) -> Self {
        Self {
            cvars: vec![ConstOrCell::Const(cst)],
            span,
        }
    }

    pub fn new_constant_typ(cst_info: &ConstInfo<F>, span: Span) -> Self {
        let ConstInfo { value, typ: _ } = cst_info;
        let cvars = value.iter().copied().map(ConstOrCell::Const).collect();

        Self { cvars, span }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cvars.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty()
    }

    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&ConstOrCell<F, C>> {
        if idx < self.cvars.len() {
            Some(&self.cvars[idx])
        } else {
            None
        }
    }

    #[must_use]
    pub fn constant(&self) -> Option<F> {
        if self.cvars.len() == 1 {
            self.cvars[0].cst()
        } else {
            None
        }
    }

    #[must_use]
    pub fn range(&self, start: usize, len: usize) -> &[ConstOrCell<F, C>] {
        &self.cvars[start..(start + len)]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ConstOrCell<F, C>> {
        self.cvars.iter()
    }
}

// implement indexing into Var
impl<F: Field, C: BackendVar> std::ops::Index<usize> for Var<F, C> {
    type Output = ConstOrCell<F, C>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cvars[index]
    }
}

/// Represents a variable in the circuit, or a reference to one.
/// Note that mutable variables are always passed as references,
/// as one needs to have access to the variable name to be able to reassign it in the environment.
pub enum VarOrRef<B>
where
    B: Backend,
{
    /// A [Var].
    Var(Var<B::Field, B::Var>),

    /// A reference to a noname variable in the environment.
    /// Potentially narrowing it down to a range of cells in that variable.
    /// For example, `x[2]` would be represented with "x" and the range `(2, 1)`,
    /// if `x` is an array of `Field` elements.
    Ref {
        var_name: String,
        start: usize,
        len: usize,
    },
}

impl<B: Backend> VarOrRef<B> {
    pub(crate) fn constant(&self) -> Option<B::Field> {
        match self {
            VarOrRef::Var(var) => var.constant(),
            VarOrRef::Ref { .. } => None,
        }
    }

    /// Returns the value within the variable or pointer.
    /// If it is a pointer, we lose information about the original variable,
    /// thus calling this function is aking to passing the variable by value.
    pub(crate) fn value(
        self,
        circuit_writer: &CircuitWriter<B>,
        fn_env: &FnEnv<B::Field, B::Var>,
    ) -> Var<B::Field, B::Var> {
        match self {
            VarOrRef::Var(var) => var,
            VarOrRef::Ref {
                var_name,
                start,
                len,
            } => {
                let var_info = circuit_writer.get_local_var(fn_env, &var_name);
                let cvars = var_info.var.range(start, len).to_vec();
                Var::new(cvars, var_info.var.span)
            }
        }
    }

    pub(crate) fn from_var_info(var_name: String, var_info: VarInfo<B::Field, B::Var>) -> Self {
        if var_info.mutable {
            Self::Ref {
                var_name,
                start: 0,
                len: var_info.var.len(),
            }
        } else {
            Self::Var(var_info.var)
        }
    }

    pub(crate) fn narrow(&self, start: usize, len: usize) -> Self {
        match self {
            VarOrRef::Var(var) => {
                let cvars = var.range(start, len).to_vec();
                VarOrRef::Var(Var::new(cvars, var.span))
            }

            //      old_start
            //      |
            //      v
            // |----[-----------]-----| <-- var.cvars
            //       <--------->
            //         old_len
            //
            //
            //          start
            //          |
            //          v
            //      |---[-----]-|
            //           <--->
            //            len
            //
            VarOrRef::Ref {
                var_name,
                start: old_start,
                len: old_len,
            } => {
                // ensure that the new range is contained in the older range
                assert!(start < *old_len); // lower bound
                assert!(start + len <= *old_len); // upper bound
                assert!(len > 0); // empty range not allowed

                Self::Ref {
                    var_name: var_name.clone(),
                    start: old_start + start,
                    len,
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            VarOrRef::Var(var) => var.len(),
            VarOrRef::Ref { len, .. } => *len,
        }
    }
}

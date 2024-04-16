use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, FnEnv, VarInfo},
    constants::Span,
    error::Result,
    type_checker::ConstInfo,
    witness::{CompiledCircuit, WitnessEnv},
};

/// An internal variable that relates to a specific cell (of the execution trace),
/// or multiple cells (if wired), in the circuit.
///
/// Note: a [CellVar] is potentially not directly added to the rows,
/// for example a private input is converted directly to a (number of) [CellVar],
/// but only added to the rows when it appears in a constraint for the first time.
///
/// As the final step of the compilation,
/// we double check that all cellvars have appeared in the rows at some point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellVar {
    pub index: usize,
    pub span: Span,
}

impl CellVar {
    pub fn new(index: usize, span: Span) -> Self {
        Self { index, span }
    }
}

/// The signature of a hint function
pub type HintFn<B: Backend> = dyn Fn(&CompiledCircuit<B>, &mut WitnessEnv<B::Field>) -> Result<B::Field>;

/// A variable's actual value in the witness can be computed in different ways.
#[derive(Clone, Serialize, Deserialize)]
pub enum Value<B>
where
    B: Backend,
{
    /// Either it's a hint and can be computed from the outside.
    #[serde(skip)]
    // TODO: outch, remove hints? or https://docs.rs/serde_closure/latest/serde_closure/ ?
    Hint(Box<HintFn<B>>),

    /// Or it's a constant (for example, I wrote `2` in the code).
    #[serde(skip)]
    Constant(B::Field),

    /// Or it's a linear combination of internal circuit variables (+ a constant).
    // TODO: probably values of internal variables should be cached somewhere
    #[serde(skip)]
    LinearCombination(Vec<(B::Field, CellVar)>, B::Field /* cst */),

    Mul(CellVar, CellVar),

    #[serde(skip)]
    Scale(B::Field, CellVar),

    /// Returns the inverse of the given variable.
    /// Note that it will potentially return 0 if the given variable is 0.
    Inverse(CellVar),

    /// A public or private input to the function
    /// There's an index associated to a variable name, as the variable could be composed of several field elements.
    External(String, usize),

    /// A public output.
    /// This is tracked separately as public inputs as it needs to be computed later.
    PublicOutput(Option<CellVar>),
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
pub enum ConstOrCell<F>
where
    F: Field,
{
    /// A constant value.
    #[serde(skip)]
    Const(F),

    /// A cell in the execution trace.
    Cell(CellVar),
}

impl<F: Field> ConstOrCell<F> {
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    pub fn cst(&self) -> Option<F> {
        match self {
            Self::Const(cst) => Some(*cst),
            _ => None,
        }
    }

    pub fn cvar(&self) -> Option<&CellVar> {
        match self {
            Self::Cell(cvar) => Some(cvar),
            _ => None,
        }
    }

    pub fn idx(&self) -> Option<usize> {
        match self {
            Self::Cell(cell) => Some(cell.index),
            _ => None,
        }
    }
}

/// Represents a variable in the noname language, or an anonymous variable during computation of expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Var<F>
where
    F: Field,
{
    /// The type of variable.
    pub cvars: Vec<ConstOrCell<F>>,

    /// The span that created the variable.
    pub span: Span,
}

impl<F: Field> Var<F> {
    pub fn new(cvars: Vec<ConstOrCell<F>>, span: Span) -> Self {
        Self { cvars, span }
    }

    pub fn new_cvar(cvar: ConstOrCell<F>, span: Span) -> Self {
        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_var(cvar: CellVar, span: Span) -> Self {
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
        let cvars = value.into_iter().cloned().map(ConstOrCell::Const).collect();

        Self { cvars, span }
    }

    pub fn len(&self) -> usize {
        self.cvars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&ConstOrCell<F>> {
        if idx < self.cvars.len() {
            Some(&self.cvars[idx])
        } else {
            None
        }
    }

    pub fn constant(&self) -> Option<F> {
        if self.cvars.len() == 1 {
            self.cvars[0].cst()
        } else {
            None
        }
    }

    pub fn range(&self, start: usize, len: usize) -> &[ConstOrCell<F>] {
        &self.cvars[start..(start + len)]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ConstOrCell<F>> {
        self.cvars.iter()
    }
}

// implement indexing into Var
impl<F: Field> std::ops::Index<usize> for Var<F> {
    type Output = ConstOrCell<F>;

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
    Var(Var<B::Field>),

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
        fn_env: &FnEnv<B::Field>,
    ) -> Var<B::Field> {
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

    pub(crate) fn from_var_info(var_name: String, var_info: VarInfo<B::Field>) -> Self {
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

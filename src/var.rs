use std::vec;

use crate::{
    constants::{Field, Span},
    error::Result,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
pub type HintFn = dyn Fn(&CompiledCircuit, &mut WitnessEnv) -> Result<Field>;

/// A variable's actual value in the witness can be computed in different ways.
pub enum Value {
    /// Either it's a hint and can be computed from the outside.
    Hint(Box<HintFn>),

    /// Or it's a constant (for example, I wrote `2` in the code).
    Constant(Field),

    /// Or it's a linear combination of internal circuit variables (+ a constant).
    // TODO: probably values of internal variables should be cached somewhere
    LinearCombination(Vec<(Field, CellVar)>, Field /* cst */),

    Mul(CellVar, CellVar),

    Scale(Field, CellVar),

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

impl From<Field> for Value {
    fn from(field: Field) -> Self {
        Self::Constant(field)
    }
}

impl std::fmt::Debug for Value {
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
#[derive(Debug, Clone, Copy)]
pub enum ConstOrCell {
    /// A constant value.
    Const(Field),

    /// A cell in the execution trace.
    Cell(CellVar),
}

impl ConstOrCell {
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    pub fn cst(&self) -> Option<Field> {
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
#[derive(Debug, Clone)]
pub struct Var {
    /// The type of variable.
    pub cvars: Vec<ConstOrCell>,

    /// The span that created the variable.
    pub span: Span,
}

impl Var {
    pub fn new(cvars: Vec<ConstOrCell>, span: Span) -> Self {
        Self { cvars, span }
    }

    pub fn new_cvar(cvar: ConstOrCell, span: Span) -> Self {
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

    pub fn new_constant(cst: Field, span: Span) -> Self {
        Self {
            cvars: vec![ConstOrCell::Const(cst)],
            span,
        }
    }

    pub fn len(&self) -> usize {
        self.cvars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&ConstOrCell> {
        if idx < self.cvars.len() {
            Some(&self.cvars[idx])
        } else {
            None
        }
    }

    pub fn constant(&self) -> Option<Field> {
        if self.cvars.len() == 1 {
            self.cvars[0].cst()
        } else {
            None
        }
    }

    pub fn range(&self, start: usize, len: usize) -> &[ConstOrCell] {
        &self.cvars[start..(start + len)]
    }
}

// implement indexing into Var
impl std::ops::Index<usize> for Var {
    type Output = ConstOrCell;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cvars[index]
    }
}

/// the equivalent of [CellVar]s but for witness generation
#[derive(Debug, Clone)]
pub struct CellValues {
    pub values: Vec<Field>,
}

impl CellValues {
    pub fn new(values: Vec<Field>) -> Self {
        Self { values }
    }
}

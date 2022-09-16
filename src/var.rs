use std::{collections::HashMap, vec};

use ark_ff::{One, Zero};

use crate::{
    circuit_writer::CircuitWriter,
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

impl From<usize> for Value {
    fn from(cst: usize) -> Self {
        let cst: u32 = cst
            .try_into()
            .expect("number too large (TODO: better error?)");
        Self::Constant(Field::try_from(cst).unwrap())
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
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// A constant value created in a noname program
pub struct Constant {
    /// The actual value.
    pub value: Field,

    /// The span that created the constant.
    pub span: Span,
}

impl Constant {
    pub fn new(value: Field, span: Span) -> Self {
        Self { value, span }
    }

    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    pub fn constrain(&self, label: Option<&'static str>, compiler: &mut CircuitWriter) -> CellVar {
        compiler.add_constant(label, self.value, self.span)
    }
}

/// Represents a cell in the execution trace.
#[derive(Debug, Clone)]
pub enum ConstOrCell {
    /// A constant value.
    Const(Constant),

    /// A cell in the execution trace.
    Cell(CellVar),
}

impl ConstOrCell {
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    pub fn cst(&self) -> Option<&Constant> {
        match self {
            Self::Const(cst) => Some(cst),
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

#[derive(Debug, Clone)]
/// A variable in a program can have different shapes.
pub enum VarKind {
    /// We pack [Const] and [CellVar] in the same enum because we often branch on these.
    ConstOrCell(ConstOrCell),

    /// A struct is represented as a mapping between field names and other [VarKind]s.
    Struct(HashMap<String, VarKind>),

    /// An array or a tuple is represetend as a list of other [VarKind]s.
    ArrayOrTuple(Vec<VarKind>),
}

impl VarKind {
    pub fn new_cell(cell: CellVar) -> Self {
        Self::ConstOrCell(ConstOrCell::Cell(cell))
    }

    pub fn new_constant(cst: Constant) -> Self {
        Self::ConstOrCell(ConstOrCell::Const(cst))
    }

    /// Recursively search if there's a constant value somewhere in this var.
    pub fn has_constants(&self) -> bool {
        match self {
            VarKind::ConstOrCell(c) => match c {
                ConstOrCell::Const(_) => return true,
                ConstOrCell::Cell(_) => return false,
            },
            VarKind::Struct(stru) => {
                for (_, var) in stru {
                    if var.has_constants() {
                        return true;
                    }
                }
            }
            VarKind::ArrayOrTuple(array) => {
                for var in array {
                    if var.has_constants() {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    pub fn len(&self) -> usize {
        match self {
            VarKind::ConstOrCell(_) => 1,
            VarKind::Struct(s) => {
                let mut sum = 0;
                for v in s.values() {
                    sum += v.len();
                }
                sum
            }
            VarKind::ArrayOrTuple(a) => {
                let mut sum = 0;
                for v in a {
                    sum += v.len();
                }
                sum
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn const_or_cell(&self) -> Option<&ConstOrCell> {
        match self {
            VarKind::ConstOrCell(c) => Some(c),
            _ => None,
        }
    }

    pub fn into_const_or_cells(&self) -> Vec<&ConstOrCell> {
        match self {
            VarKind::ConstOrCell(t) => vec![t],
            VarKind::Struct(stru) => {
                let mut res = vec![];
                for var in stru.values() {
                    res.extend(var.into_const_or_cells());
                }
                res
            }
            VarKind::ArrayOrTuple(array) => {
                let mut res = vec![];
                for var in array {
                    res.extend(var.into_const_or_cells());
                }
                res
            }
        }
    }
}

/// Represents a variable in the noname language, or an anonymous variable during computation of expressions.
#[derive(Debug, Clone)]
pub struct Var {
    /// The type of variable.
    pub kind: VarKind,

    /// The span that created the variable.
    pub span: Span,
}

impl Var {
    pub fn new(kind: VarKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn new_var(var: CellVar, span: Span) -> Self {
        Self {
            kind: VarKind::ConstOrCell(ConstOrCell::Cell(var)),
            span,
        }
    }

    pub fn new_constant(cst: Constant, span: Span) -> Self {
        Self {
            kind: VarKind::ConstOrCell(ConstOrCell::Const(cst)),
            span,
        }
    }

    pub fn new_struct(vars: HashMap<String, VarKind>, span: Span) -> Self {
        Self {
            kind: VarKind::Struct(vars),
            span,
        }
    }

    pub fn new_array(vars: Vec<VarKind>, span: Span) -> Self {
        Self {
            kind: VarKind::ArrayOrTuple(vars),
            span,
        }
    }

    pub fn array_or_tuple(&self) -> Option<&[VarKind]> {
        match &self.kind {
            VarKind::ArrayOrTuple(array) => Some(array),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.kind.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kind.is_empty()
    }

    pub fn has_constants(&self) -> bool {
        self.kind.has_constants()
    }

    pub fn const_or_cell(&self) -> Option<&ConstOrCell> {
        self.kind.const_or_cell()
    }

    pub fn constant(&self) -> Option<Constant> {
        match &self.kind {
            VarKind::ConstOrCell(ConstOrCell::Const(cst)) => Some(*cst),
            _ => None,
        }
    }

    pub fn fields(&self) -> Option<&HashMap<String, VarKind>> {
        match &self.kind {
            VarKind::Struct(stru) => Some(stru),
            _ => None,
        }
    }

    pub fn get(&self, idx: usize) -> Option<&VarKind> {
        match &self.kind {
            VarKind::ConstOrCell(_) => panic!("cannot index into a const or cell"),
            VarKind::Struct(_) => panic!("cannot index into a struct"),
            VarKind::ArrayOrTuple(array) => {
                if idx < array.len() {
                    Some(&array[idx])
                } else {
                    None
                }
            }
        }
    }

    pub fn into_const_or_cells(&self) -> Vec<&ConstOrCell> {
        self.kind.into_const_or_cells()
    }
}

// implement indexing into Var
impl std::ops::Index<usize> for Var {
    type Output = VarKind;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.kind {
            VarKind::ConstOrCell(_) | VarKind::Struct(..) => {
                panic!("bug in the compiler: indexing into a cell/const/struct")
            }
            VarKind::ArrayOrTuple(a) => &a[index],
        }
    }
}

/// the equivalent of [CellVars] but for witness generation
#[derive(Debug, Clone)]
pub struct CellValues {
    pub values: Vec<Field>,
}

impl CellValues {
    pub fn new(values: Vec<Field>) -> Self {
        Self { values }
    }
}

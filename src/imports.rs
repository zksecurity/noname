use std::{collections::HashMap, fmt};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::{
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    parser::types::{FnSig, FunctionDef},
    stdlib::{parse_fn_sigs, BUILTIN_FNS_DEFS},
    type_checker::{FnInfo, TypeChecker},
    var::Var,
};

#[derive(Debug)]
pub struct Module {
    pub name: String,
    pub kind: ModuleKind,
}

#[derive(Debug)]
pub enum ModuleKind {
    /// A module that contains only built-in functions.
    BuiltIn(BuiltinModule),

    /// A module that contains both built-in functions and native functions.
    Native(TypeChecker),
}

#[derive(Debug, Clone)]
pub struct BuiltinModule {
    pub functions: HashMap<String, FnInfo>,
}

/*
impl std::fmt::Debug for BuiltinModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ImportedModule {{ name: {:?}, functions: {:?}, span: {:?} }}",
            self.name,
            self.functions.keys(),
            self.span
        )
    }
}
*/

/// An actual handle to the internal function to call to resolve a built-in function call.
///
/// Note that the signature of a `FnHandle` is designed to:
/// * `&mut CircuitWriter`: take a mutable reference to the circuit writer, this is because built-ins need to be able to register new variables and add gates to the circuit
/// * `&[Var]`: take an unbounded list of variables, this is because built-ins can take any number of arguments, and different built-ins might take different types of arguments
/// * `Span`: take a span to return user-friendly errors
/// * `-> Result<Option<Var>>`: return a `Result` with an `Option` of a `Var`. This is because built-ins can return a variable, or they can return nothing. If they return nothing, then the `Option` will be `None`. If they return a variable, then the `Option` will be `Some(Var)`.
pub type FnHandle = fn(&mut CircuitWriter, &[VarInfo], Span) -> Result<Option<Var>>;

/// The different types of a noname function.
#[derive(Clone, Serialize, Deserialize)]
pub enum FnKind {
    /// A built-in is just a handle to a function written in Rust.
    #[serde(skip)]
    BuiltIn(FnSig, FnHandle),

    /// A native function is represented as an AST.
    Native(FunctionDef),
}

impl fmt::Debug for FnKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fnkind>")
    }
}

// static of built-in functions
pub static BUILTIN_FNS: Lazy<HashMap<String, FnInfo>> =
    Lazy::new(|| parse_fn_sigs(&BUILTIN_FNS_DEFS));

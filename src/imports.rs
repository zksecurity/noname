use std::{collections::HashMap, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    parser::types::{FnSig, FunctionDef},
    type_checker::{FnInfo, TypeChecker},
    var::Var,
};

#[derive(Debug)]
pub struct Module<B>
where
    B: Backend,
{
    pub name: String,
    pub kind: ModuleKind<B>,
}

#[derive(Debug)]
pub enum ModuleKind<B>
where
    B: Backend,
{
    /// A module that contains only built-in functions.
    BuiltIn(BuiltinModule<B>),

    /// A module that contains both built-in functions and native functions.
    Native(TypeChecker<B>),
}

#[derive(Debug, Clone)]
pub struct BuiltinModule<B>
where
    B: Backend,
{
    pub functions: HashMap<String, FnInfo<B>>,
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
pub type FnHandle<B> = fn(
    &mut CircuitWriter<B>,
    &[VarInfo<<B as Backend>::Field, <B as Backend>::Var>],
    Span,
) -> Result<Option<Var<<B as Backend>::Field, <B as Backend>::Var>>>;

/// The different types of a noname function.
#[derive(Clone, Serialize, Deserialize)]
pub enum FnKind<B>
where
    B: Backend,
{
    /// A built-in is just a handle to a function written in Rust.
    #[serde(skip)]
    BuiltIn(FnSig, FnHandle<B>),

    /// A native function is represented as an AST.
    Native(FunctionDef),
}

impl<B: Backend> fmt::Debug for FnKind<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fnkind>")
    }
}

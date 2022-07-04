use miette::Diagnostic;
use thiserror::Error;

use crate::{lexer::TokenKind, parser::TyKind};

#[derive(Diagnostic, Debug, Error)]
#[error("Parsing error")]
#[diagnostic()]
pub struct Error {
    #[help]
    pub error: ErrorTy,

    #[label("here")]
    pub span: (usize, usize),
}

#[derive(Error, Diagnostic, Debug)]
pub enum ErrorTy {
    #[error("invalid token")]
    InvalidToken,

    #[error("missing type")]
    MissingType,

    #[error("missing token")]
    MissingToken,

    #[error("invalid token, expected: {0}")]
    ExpectedToken(TokenKind),

    #[error("invalid path")]
    InvalidPath,

    #[error("invalid end of line")]
    InvalidEndOfLine,

    #[error("invalid module")]
    InvalidModule,

    #[error("invalid function signature: {0}")]
    InvalidFunctionSignature(&'static str),

    #[error("invalid function name")]
    InvalidFunctionName,

    #[error("invalid type name")]
    InvalidTypeName,

    #[error("invalid type, expected an array or a type name (starting with an uppercase letter, and only containing alphanumeric characters)")]
    InvalidType,

    #[error("invalid array size")]
    InvalidArraySize,

    #[error("invalid statement")]
    InvalidStatement,

    #[error("missing expression")]
    MissingExpression,

    #[error("invalid expression")]
    InvalidExpression,

    #[error("invalid identifier, expected lowercase alphanumeric string (including underscore `_`) and starting with a letter")]
    InvalidIdentifier,

    #[error("invalid function call")]
    InvalidFnCall,

    #[error("imports via `use` keyword must appear before anything else")]
    UseAfterFn,

    #[error("function {0} is not recognized")]
    UnknownFunction(String),

    #[error(
        "function {fn_name} expected {expected_args} arguments but was passed {observed_args}"
    )]
    WrongNumberOfArguments {
        fn_name: String,
        expected_args: usize,
        observed_args: usize,
    },

    #[error("argument `{arg_name}` of function {fn_name} was passed a type {observed_ty} when it expected a {expected_ty}")]
    WrongArgumentType {
        fn_name: String,
        arg_name: String,
        expected_ty: String,
        observed_ty: String,
    },

    #[error("cannot compute the expression")]
    CannotComputeExpression,

    #[error("type {0} and {1} are not compatible")]
    MismatchType(TyKind, TyKind),

    #[error("variable used is not defined anywhere")]
    UndefinedVariable,

    #[error("you cannot have a return statement in the `main` function")]
    ReturnInMain,

    #[error("unexpected argument type in function call. Expected: {0} and got {1}")]
    ArgumentTypeMismatch(TyKind, TyKind),

    #[error("the function `{0}` return value must be used")]
    FunctionReturnsType(String),

    #[error("missing argument `{0}`")]
    MissingArg(String),

    #[error("cannot convert `{0}` to field element")]
    CannotConvertToField(String),
}

use miette::Diagnostic;
use thiserror::Error;

use crate::lexer::TokenKind;

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

    #[error("function {name} expected {expected_args} arguments but was passed {observed_args}")]
    WrongNumberOfArguments {
        name: String,
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
}

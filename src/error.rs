use miette::Diagnostic;
use thiserror::Error;

use crate::lexer::TokenType;

#[derive(Diagnostic, Debug, Error)]
#[error("oops")]
#[diagnostic(code(my_lib::random_error))]
pub struct Error {
    #[help]
    pub error: ErrorTy,

    #[label("here")]
    pub span: (usize, usize),
}

#[derive(Error, Diagnostic, Debug)]
pub enum ErrorTy {
    #[error("invalid token")]
    #[diagnostic(code(my_lib::io_error))]
    InvalidToken,

    #[error("missing type")]
    #[diagnostic(code(my_lib::io_error))]
    MissingType,

    #[error("missing token")]
    #[diagnostic(code(my_lib::io_error))]
    MissingToken,

    #[error("invalid token, expected: {0}")]
    #[diagnostic(code(my_lib::io_error))]
    ExpectedToken(TokenType),

    #[error("invalid path")]
    #[diagnostic(code(my_lib::io_error))]
    InvalidPath,

    #[error("invalid end of line")]
    #[diagnostic(code(my_lib::io_error))]
    InvalidEndOfLine,

    #[error("invalid module")]
    InvalidModule,

    #[error("invalid function signature")]
    InvalidFunctionSignature,

    #[error("invalid function name")]
    InvalidFunctionName,

    #[error("invalid type name")]
    InvalidTypeName,

    #[error("invalid type")]
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
}

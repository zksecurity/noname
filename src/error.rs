use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

#[derive(Diagnostic, Debug, Error)]
#[error("oops")]
#[diagnostic(code(my_lib::random_error))]
pub struct Error {
    pub error: ErrorTy,

    #[label("here")]
    pub span: (usize, usize),
}

#[derive(Error, Diagnostic, Debug)]
pub enum ErrorTy {
    #[error("invalid token")]
    #[diagnostic(code(my_lib::io_error))]
    InvalidToken,

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
}

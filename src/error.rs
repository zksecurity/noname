use miette::Diagnostic;
use thiserror::Error;

use crate::{
    lexer::TokenKind,
    parser::{AttributeKind, TyKind},
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Diagnostic, Debug, Error)]
#[error("I'm so sorry, looks like something went wrong...")]
#[diagnostic()]
pub struct Error {
    #[help]
    pub kind: ErrorKind,

    #[label("here")]
    pub span: (usize, usize),
}

#[derive(Error, Diagnostic, Debug)]
pub enum ErrorKind {
    #[error("error used for tests only")]
    TestError,

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

    #[error("invalid function call: {0}")]
    InvalidFnCall(&'static str),

    #[error("imports via `use` keyword must appear before anything else")]
    UseAfterFn,

    #[error("function expected {expected_args} arguments but was passed {observed_args}")]
    WrongNumberOfArguments {
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

    #[error("unexpected argument type in function call. Expected: {0} and got {1}")]
    ArgumentTypeMismatch(TyKind, TyKind),

    #[error("the function `{0}` return value must be used")]
    FunctionReturnsType(String),

    #[error("missing argument `{0}`")]
    MissingArg(String),

    #[error("cannot convert `{0}` to field element")]
    CannotConvertToField(String),

    #[error("return value is `{0}` when `{1}` was expected")]
    ReturnTypeMismatch(TyKind, TyKind),

    #[error("public output not set as part of the circuit")]
    MissingPublicOutput,

    #[error("missing public output type in the function signature")]
    NoPublicOutput,

    #[error("error while importing std path: {0}")]
    StdImport(&'static str),

    #[error("tried to import the same module `{0}` twice")]
    DuplicateModule(String),

    #[error("`public_output` is a reserved argument name")]
    PublicOutputReserved,

    #[error("function `{0}` not present in scope")]
    UndefinedFunction(String),

    #[error("module `{0}` not present in scope")]
    UndefinedModule(String),

    #[error("attribute not recognized: `{0:?}`")]
    InvalidAttribute(AttributeKind),

    #[error("expressions that return a value are forbidden as statements")]
    ExpectedUnitExpr,

    #[error("array accessed at index {0} is out of bounds (max {1})")]
    ArrayIndexOutOfBounds(usize, usize),

    #[error("array indexes must be constants in circuits")]
    ExpectedConstant,

    #[error("kimchi setup: {0}")]
    KimchiSetup(#[from] kimchi::error::SetupError),

    #[error("kimchi prover: {0}")]
    KimchiProver(#[from] kimchi::error::ProverError),

    #[error("kimchi verifier: {0}")]
    KimchiVerifier(#[from] kimchi::error::VerifyError),

    #[error("invalid witness (row {0})")]
    InvalidWitness(usize),

    #[error("user provided input `{0}` is not defined in the main function's arguments")]
    UnusedInput(String),

    #[error("private input not used in the circuit")]
    PrivateInputNotUsed,
}

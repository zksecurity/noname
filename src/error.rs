use miette::Diagnostic;
use thiserror::Error;

use crate::{
    constants::Span,
    lexer::TokenKind,
    parser::{AttributeKind, TyKind},
};

pub type Result<T> = std::result::Result<T, Error>;

/// An error in noname.
#[derive(Diagnostic, Debug, Error)]
#[error("Looks like something went wrong...")]
pub struct Error {
    /// The type of error.
    #[help]
    pub kind: ErrorKind,

    /// Indicate where the error occurred in the source code.
    #[label("here")]
    pub span: Span,
}

impl Error {
    /// Creates a new [Error] from an [ErrorKind].
    pub fn new(kind: ErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// The type of error.
#[derive(Error, Diagnostic, Debug, Clone)]
pub enum ErrorKind {
    #[error("variable is not mutable. You must set the `mut` keyword to make it mutable")]
    AssignmentToImmutableVariable,

    #[error("the `const` attribute cannot be used for arguments of the main function")]
    ConstArgumentNotForMain,

    #[error("a field access or a method call can only be applied on a field of another struct, a struct, or an array access")]
    InvalidFieldAccessExpression,

    #[error("the method called is not a static method")]
    NotAStaticMethod,

    #[error("{0} arguments are passed when {1} were expected")]
    MismatchFunctionArguments(usize, usize),

    #[error("the `pub` keyword is reserved for arguments of the main function")]
    PubArgumentOutsideMain,

    #[error("the function main is not recursive")]
    RecursiveMain,

    #[error("invalid token")]
    InvalidToken,

    #[error("missing type")]
    MissingType,

    #[error("missing token")]
    MissingToken,

    #[error("invalid token, expected: {0}")]
    ExpectedToken(TokenKind),

    #[error("invalid path: {0}")]
    InvalidPath(&'static str),

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

    #[error("the custom type name used: `{0}` is a reserved type name")]
    ReservedType(String),

    #[error("invalid array size, expected [_; x] with x in [0,2^32]")]
    InvalidArraySize,

    #[error("the value passed could not be converted to a field element")]
    InvalidField(String),

    #[error("invalid range size, expected x..y with x and y integers in [0,2^32]")]
    InvalidRangeSize,

    #[error("invalid statement")]
    InvalidStatement,

    #[error("missing expression")]
    MissingExpression,

    #[error("invalid expression")]
    InvalidExpression,

    #[error("invalid identifier `{0}`, expected lowercase alphanumeric string (including underscore `_`) and starting with a letter")]
    InvalidIdentifier(String),

    #[error("invalid function call: {0}")]
    InvalidFnCall(&'static str),

    #[error("imports via `use` keyword must appear before anything else")]
    UseAfterFn,

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

    #[error("you need to pass the following argument: `{0}`")]
    MissingArg(String),

    #[error("cannot convert `{0}` to field element")]
    CannotConvertToField(String),

    #[error("return value is `{0}` when `{1}` was expected")]
    ReturnTypeMismatch(TyKind, TyKind),

    #[error("a return value was expected by the function signature")]
    MissingReturn,

    #[error("public output not set as part of the circuit")]
    MissingPublicOutput,

    #[error("missing return type in the function signature")]
    UnexpectedReturn,

    #[error("error while importing std path: {0}")]
    StdImport(&'static str),

    #[error("tried to import the same module `{0}` twice")]
    DuplicateModule(String),

    #[error("`{0}` is a reserved argument name")]
    PublicOutputReserved(String),

    #[error("function `{0}` not present in scope")]
    UndefinedFunction(String),

    #[error("module `{0}` not present in scope")]
    UndefinedModule(String),

    #[error("attribute not recognized: `{0:?}`")]
    InvalidAttribute(AttributeKind),

    #[error("A return value is not used")]
    UnusedReturnValue,

    #[error("array accessed at index {0} is out of bounds (max {1})")]
    ArrayIndexOutOfBounds(usize, usize),

    #[error(
        "one-letter variables or types are not allowed. Best practice is to use descriptive names"
    )]
    NoOneLetterVariable,

    #[error("array indexes must be constants in circuits")]
    ExpectedConstant,

    #[error("kimchi setup: {0}")]
    KimchiSetup(#[from] kimchi::error::SetupError),

    #[error("kimchi prover: {0}")]
    KimchiProver(#[from] kimchi::error::ProverError),

    #[error("kimchi verifier: {0}")]
    KimchiVerifier(#[from] kimchi::error::VerifyError),

    #[error("the program did not run to completion with the given private and/or public inputs (row {0} of the witness failed to verify)")]
    InvalidWitness(usize),

    #[error("user provided input `{0}` is not defined in the main function's arguments")]
    UnusedInput(String),

    #[error("private input not used in the circuit")]
    PrivateInputNotUsed,

    #[error("the variable `{0}` is declared twice")]
    DuplicateDefinition(String),

    #[error("only variables and arrays can be mutated")]
    InvalidAssignmentExpression,

    #[error("the main function must have at least one argument")]
    NoArgsInMain,

    #[error("local variable `{0}` couldn't be found")]
    LocalVariableNotFound(String),

    #[error("the public output cannot contain constants")]
    ConstantInOutput,

    #[error("incorrect number of fields declared for the `{0}` struct declaration")]
    MismatchStructFields(String),

    #[error("invalid field, expected `{0}` and got `{1}`")]
    InvalidStructField(String, String),

    #[error("invalid type for the field, expected `{0}` and got `{1}`")]
    InvalidStructFieldType(TyKind, TyKind),

    #[error("struct `{0}` is never defined")]
    UndefinedStruct(String),

    #[error("struct `{0}` does not have a field called `{1}`")]
    UndefinedField(String, String),

    #[error("this assertion failed")]
    AssertionFailed,

    #[error("constants can only have a literal decimal value")]
    InvalidConstType,

    #[error("cannot compile a module without a main function")]
    NoMainFunction,
}

use clap::error;
use miette::Diagnostic;
use thiserror::Error;

use crate::{
    constants::Span,
    inputs::ParsingError,
    lexer::TokenKind,
    parser::types::{AttributeKind, TyKind},
};

pub type Result<T> = std::result::Result<T, Error>;

/// An error in noname.
#[derive(Diagnostic, Debug, Error)]
#[error("Looks like something went wrong in {label}")]
pub struct Error {
    /// A hint as to where the error happened (e.g. type-checker, parser, lexer, etc.)
    pub label: &'static str,

    /// The type of error.
    #[help]
    pub kind: ErrorKind,

    /// Indicate where the error occurred in the source code.
    #[label("here")]
    pub span: Span,
}

impl Error {
    /// Creates a new [Error] from an [ErrorKind].
    pub fn new(label: &'static str, kind: ErrorKind, span: Span) -> Self {
        Self { label, kind, span }
    }
}

/// The type of error.
#[derive(Error, Diagnostic, Debug)]
pub enum ErrorKind {
    #[error(
        "Unexpected error: {0}. Please report this error on https://github.com/mimoo/noname/issues"
    )]
    UnexpectedError(&'static str),
    #[error("variable is not mutable. You must set the `mut` keyword to make it mutable")]
    AssignmentToImmutableVariable,
    #[error("the {0} of assert_eq must be of type Field or BigInt. It was of type {1}")]
    AssertTypeMismatch(&'static str, TyKind),
    #[error("the types in assert_eq don't match: expected {0} but got {1}")]
    AssertEqTypeMismatch(TyKind, TyKind),
    #[error(
        "the dependency `{0}` does not appear to be listed in your manifest file `Noname.toml`"
    )]
    UnknownDependency(String),
    #[error("the function `{1}` does not exist in the module `{0}`")]
    UnknownExternalFn(String, String),

    #[error("the struct `{1}` does not exist in the module `{0}`")]
    UnknownExternalStruct(String, String),

    #[error("you cannot have a function called `main` in a library")]
    MainFunctionInLib,

    #[error("you cannot call your function `{0}`, as it already exists as a builtin function (try renaming your function)")]
    ShadowingBuiltIn(String),

    #[error(transparent)]
    ParsingError(#[from] ParsingError),

    #[error("the `const` attribute cannot be used for arguments of the main function")]
    ConstArgumentNotForMain,

    #[error("a field access or a method call can only be applied on a field of another struct, a struct, or an array access")]
    InvalidFieldAccessExpression,

    #[error("the method called is not a static method")]
    NotAStaticMethod,

    #[error("{0} arguments are passed when {1} were expected")]
    MismatchFunctionArguments(usize, usize),

    #[error("constants must be declared before any structs or functions")]
    OrderOfConstDeclaration,

    #[error(
        "the `use` keyword must be used before anything else (consts, structs, functions, etc.)"
    )]
    OrderOfUseDeclaration,

    #[error("cannot chain arithmetic operations without using parenthesis")]
    MissingParenthesis,

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

    #[error("invalid token, unexpected keyword: {0}, expected: {1}")]
    ExpectedTokenNotKeyword(String, TokenKind),

    #[error("invalid path: {0}")]
    InvalidPath(&'static str),

    #[error("invalid end of line")]
    InvalidEndOfLine,

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

    #[error("Invalid expression in symbolic size")]
    InvalidSymbolicSize,

    #[error("invalid generic parameter, expected single uppercase letter, such as N, M, etc.")]
    InvalidGenericParameter,

    #[error("missing generic value: `{0}`")]
    GenericValueExpected(String),

    #[error("conflict generic values during binding for `{0}`: `{1}` and `{2}`")]
    ConflictGenericValue(String, String, String),

    #[error("unexpected generic parameter: `{0}`")]
    UnexpectedGenericParameter(String),

    #[error("calling generic functions in for loop is not allowed")]
    GenericInForLoop,

    #[error("variable `{0}` is forbidden to be accessed in the for loop, likely due to a generic function call")]
    VarAccessForbiddenInForLoop(String),

    #[error("the value passed could not be converted to a field element")]
    InvalidField(String),

    #[error("invalid range size, expected x..y with x and y integers in [0,2^32]")]
    InvalidRangeSize,

    #[error("invalid iterator type, expected an array but found {0}")]
    InvalidIteratorType(TyKind),

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

    #[error("type '{0}' and '{1}' are not compatible")]
    MismatchType(TyKind, TyKind),

    #[error("variable used is not defined anywhere")]
    UndefinedVariable,

    #[error("unexpected argument type in function call. Expected: {0} and got {1}")]
    ArgumentTypeMismatch(TyKind, TyKind),

    #[error("the function `{0}` return value must be used")]
    FunctionReturnsType(String),

    #[error("you need to pass the following public argument: `{0}`")]
    MissingPublicArg(String),

    #[error("you need to pass the following private argument: `{0}`")]
    MissingPrivateArg(String),

    #[error("cannot convert `{0}` to field element")]
    CannotConvertToField(String),

    #[error("cannot convert `{0}` to {1}")]
    CannotConvertFromField(String, String),

    #[error("a return value was expected by the function signature")]
    MissingReturn,

    #[error("no return value was expected as part of this function signature")]
    NoReturnExpected,

    #[error("the `self` argument cannot have attributes")]
    SelfHasAttribute,

    #[error("the return type observed (`{0}`) doesn't match what the function expected as return type (`{1}`)")]
    ReturnTypeMismatch(TyKind, TyKind),

    #[error("missing return type in the function signature")]
    UnexpectedReturn,

    #[error("error while importing std path: {0}")]
    StdImport(String),

    #[error("tried to import the same module `{0}` twice")]
    DuplicateModule(String),

    #[error("`{0}` is a reserved argument name")]
    PublicOutputReserved(String),

    #[error("function `{0}` not present in scope (did you misspell it?)")]
    UndefinedFunction(String),

    #[error("hint function `{0}` signature is missing its corresponding builtin function")]
    MissingHintMapping(String),

    #[error("function name `{0}` is already in use by a variable present in the scope")]
    FunctionNameInUsebyVariable(String),

    #[error("module `{0}` not present in scope (are you sure you imported it?)")]
    UndefinedModule(String),

    #[error("attribute not recognized: `{0:?}`")]
    InvalidAttribute(AttributeKind),

    #[error("unsafe attribute is needed to call a hint function. eg: `unsafe fn foo()`")]
    ExpectedUnsafeAttribute,

    #[error("unsafe attribute should only be applied to hint function calls")]
    UnexpectedUnsafeAttribute,

    #[error("A return value is not used")]
    UnusedReturnValue,

    #[error("array accessed at index {0} is out of bounds (max allowed index is {1})")]
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

    #[error("unexpected assignment type. Expected: {0} and got {1}")]
    AssignmentTypeMismatch(TyKind, TyKind),

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

    #[error("method call can only be applied on custom structs")]
    MethodCallOnNonCustomStruct,

    #[error("field access can only be applied on custom structs")]
    FieldAccessOnNonCustomStruct,

    #[error("array like access can only be performed on arrays or tuples")]
    AccessOnNonCollection,

    #[error("struct `{0}` does not exist (are you sure it is defined?)")]
    UndefinedStruct(String),

    #[error("struct `{0}` does not have a method called `{1}`")]
    UndefinedMethod(String, String),

    #[error("struct `{0}` does not have a field called `{1}`")]
    UndefinedField(String, String),

    #[error("this assertion failed")]
    AssertionFailed,

    #[error("constants can only have a literal decimal value")]
    InvalidConstType,

    #[error("cannot compile a module without a main function")]
    NoMainFunction,

    #[error("invalid hexadecimal literal `${0}`")]
    InvalidHexLiteral(String),

    #[error("if-else condition is expected to be a boolean, but has type `{0}`")]
    IfElseInvalidConditionType(TyKind),

    #[error("`if` branch must be a variable, a field access, or an array access. It can't be logic that creates constraints.")]
    IfElseInvalidIfBranch(),

    #[error("`else` branch must be a variable, a field access, or an array access. It can't be logic that creates constraints.")]
    IfElseInvalidElseBranch(),

    #[error("`if` branch and `else` branch must have matching types")]
    IfElseMismatchingBranchesTypes(),

    #[error("invalid range, the end value can't be smaller than the start value")]
    InvalidRange,

    #[error("division by zero")]
    DivisionByZero,

    #[error("Not enough variables provided to fill placeholders in the formatted string")]
    InsufficientVariables,
}

use crate::{
    circuit_writer::CircuitWriter,
    error::Result,
    lexer::Token,
    parser::AST,
    type_checker::{Dependencies, TypeChecker},
    witness::CompiledCircuit,
};

pub fn compile_single(filename: String, code: String) -> Result<CompiledCircuit> {
    let tast = get_tast_single(filename, code)?;
    CircuitWriter::generate_circuit(tast, Dependencies::default())
}

pub fn get_tast_single(filename: String, code: String) -> Result<TypeChecker> {
    get_tast(filename, code, &Dependencies::default())
}

pub fn get_tast(filename: String, code: String, deps: &Dependencies) -> Result<TypeChecker> {
    let tokens = Token::parse(&filename, &code)?;
    let ast = AST::parse(&filename, &code, tokens)?;
    TypeChecker::analyze(filename, code, ast, deps)
}

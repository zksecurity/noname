use crate::{
    circuit_writer::CircuitWriter,
    error::Result,
    lexer::Token,
    parser::AST,
    type_checker::{Dependencies, TAST},
    witness::CompiledCircuit,
};

pub fn compile(code: &str, deps: &Dependencies) -> Result<CompiledCircuit> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // TAST
    let tast = TAST::analyze(ast, deps)?;

    // type checker + compiler
    let circuit = CircuitWriter::generate_circuit(tast, deps, code)?;

    //
    Ok(circuit)
}

pub fn get_tast(code: &str, deps: &Dependencies) -> miette::Result<TAST> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // TAST
    let tast = TAST::analyze(ast, deps)?;

    //
    Ok(tast)
}

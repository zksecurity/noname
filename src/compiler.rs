use crate::{
    circuit_writer::CircuitWriter,
    error::Result,
    lexer::Token,
    parser::AST,
    type_checker::{Dependencies, TypeChecker},
    witness::CompiledCircuit,
};

pub fn compile(code: &str, deps: Dependencies) -> Result<CompiledCircuit> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // TAST
    let tast = TypeChecker::analyze(ast, &deps)?;

    // type checker + compiler
    let circuit = CircuitWriter::generate_circuit(tast, deps, code)?;

    //
    Ok(circuit)
}

pub fn get_tast(code: &str, deps: &Dependencies) -> miette::Result<TypeChecker> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // TAST
    let tast = TypeChecker::analyze(ast, deps)?;

    //
    Ok(tast)
}

use crate::{
    circuit_writer::CircuitWriter, error::Result, lexer::Token, parser::AST, type_checker::TAST,
    witness::CompiledCircuit,
};

pub fn compile(code: &str) -> Result<CompiledCircuit> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // TAST
    let tast = TAST::analyze(ast)?;

    // type checker + compiler
    let circuit = CircuitWriter::generate_circuit(tast, code)?;

    //
    Ok(circuit)
}

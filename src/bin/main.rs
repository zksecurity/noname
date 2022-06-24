use miette::{IntoDiagnostic, Result, WrapErr};
use my_programming_language::{lexer::Token, parser::AST};

fn parse(code: &str) -> Result<AST> {
    let tokens = Token::parse(code)?;
    let ast = AST::parse(tokens)?;
    Ok(ast)
}

fn main() -> Result<()> {
    let code = std::fs::read_to_string("data/poseidon.no")
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    parse(&code).map_err(|e| e.with_source_code(code))?;

    Ok(())
}

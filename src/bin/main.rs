use std::{collections::HashMap, path::PathBuf};

use clap::Parser;
use miette::{IntoDiagnostic, Result, WrapErr};
use my_programming_language::{
    ast::{Compiler, F},
    lexer::Token,
    parser::AST,
};

fn parse(code: &str) -> Result<()> {
    let tokens = Token::parse(code)?;
    let ast = AST::parse(tokens)?;
    let (circuit, compiler) = Compiler::analyze_and_compile(ast, code)?;

    println!("circuit: {circuit}");

    let mut args = HashMap::new();
    args.insert("public_input", F::one());
    args.insert("private_input", F::one());
    let witness = compiler.generate_witness(args);

    Ok(())
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser)]
    path: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let code = std::fs::read_to_string(cli.path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    parse(&code).map_err(|e| e.with_source_code(code))?;

    println!("successfuly compiled");

    Ok(())
}

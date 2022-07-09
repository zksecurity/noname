use std::{collections::HashMap, path::PathBuf};

use ark_ff::One;
use clap::Parser;
use miette::{IntoDiagnostic, Result, WrapErr};
use my_programming_language::{
    ast::{CircuitValue, Compiler, Gate},
    constants::IO_REGISTERS,
    field::Field,
    lexer::Token,
    parser::AST,
    prover::compile,
};

fn parse(code: &str, debug: bool) -> Result<()> {
    // compile
    let (circuit, prover_index, verifier_index) = compile(code, debug)?;

    // print ASM
    println!("{circuit}");

    // generate witness
    let mut args = HashMap::new();
    args.insert("public_input", CircuitValue::new(vec![Field::one()]));
    args.insert(
        "private_input",
        CircuitValue::new(vec![Field::one(), Field::one(), Field::one()]),
    );

    // create proof
    let (proof, full_public_inputs, public_output) = prover_index.prove(args, debug)?;

    // verify proof
    verifier_index.verify(full_public_inputs, proof)?;

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

    let code = std::fs::read_to_string(&cli.path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    let debug = true;
    parse(&code, debug).map_err(|e| e.with_source_code(code))?;

    println!("successfuly compiled");

    Ok(())
}

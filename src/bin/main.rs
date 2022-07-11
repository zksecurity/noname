use std::{collections::HashMap, path::PathBuf};

use ark_ff::One;
use clap::Parser;
use kimchi::oracle::{constants::PlonkSpongeConstantsKimchi, poseidon::Sponge};
use miette::{IntoDiagnostic, Result, WrapErr};
use my_programming_language::{
    ast::{CellValues, Compiler, Gate},
    constants::NUM_REGISTERS,
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
    args.insert("public_input", CellValues::new(vec![1.into()]));
    args.insert("private_input", CellValues::new(vec![1.into()]));

    // create proof
    let (proof, full_public_inputs, _public_output) = prover_index.prove(args, debug)?;

    // verify proof
    verifier_index.verify(full_public_inputs, proof)?;

    Ok(())
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser)]
    path: PathBuf,

    // default to false
    #[clap(short, long)]
    debug: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let code = std::fs::read_to_string(&cli.path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    parse(&code, cli.debug).map_err(|e| e.with_source_code(code))?;

    println!("successfuly compiled");

    Ok(())
}

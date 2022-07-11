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
    let private_input = vec![Field::one(), Field::one()];
    let digest = {
        let mut s: kimchi::oracle::poseidon::ArithmeticSponge<Field, PlonkSpongeConstantsKimchi> =
            kimchi::oracle::poseidon::ArithmeticSponge::new(
                kimchi::oracle::pasta::fp_kimchi::params(),
            );
        s.absorb(&private_input);
        s.squeeze()
    };

    let mut args = HashMap::new();
    args.insert("public_input", CellValues::new(vec![digest]));
    args.insert("private_input", CellValues::new(private_input));

    // create proof
    let (proof, full_public_inputs, public_output) = prover_index.prove(args, debug)?;
    //    assert_eq!(public_output.len(), 0);

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

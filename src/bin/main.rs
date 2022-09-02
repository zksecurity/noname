use std::path::PathBuf;

use clap::Parser;
use miette::{IntoDiagnostic, Result, WrapErr};
use noname::{
    inputs::{parse_inputs, Inputs},
    prover::compile,
};

fn parse(code: &str, public_inputs: Inputs, private_inputs: Inputs, debug: bool) -> Result<()> {
    // compile
    let (circuit, prover_index, verifier_index) = compile(code, debug)?;

    // print ASM
    println!("{circuit}");

    // create proof
    let (proof, full_public_inputs, _public_output) =
        prover_index.prove(public_inputs, private_inputs, debug)?;

    // verify proof
    verifier_index.verify(full_public_inputs, proof)?;

    Ok(())
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// path to the .no file
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// public inputs in a JSON format using decimal values (e.g. {"a": "1", "b": "2"})
    #[clap(long)]
    public_inputs: Option<String>,

    /// private inputs in a JSON format using decimal values (e.g. {"a": "1", "b": "2"})
    #[clap(long)]
    private_inputs: Option<String>,

    /// prints debug information (defaults to false)
    #[clap(short, long)]
    debug: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let code = std::fs::read_to_string(&cli.path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    let public_inputs = if let Some(s) = cli.public_inputs {
        parse_inputs(&s)?
    } else {
        Inputs::default()
    };

    let private_inputs = if let Some(s) = cli.private_inputs {
        parse_inputs(&s)?
    } else {
        Inputs::default()
    };

    parse(&code, public_inputs, private_inputs, cli.debug).map_err(|e| e.with_source_code(code))?;

    println!("successfuly compiled");

    Ok(())
}

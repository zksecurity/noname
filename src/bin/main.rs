use std::path::PathBuf;

use clap::Parser as _;
use miette::{IntoDiagnostic, Result, WrapErr};
use noname::{
    inputs::{parse_inputs, JsonInputs},
    prover::compile_and_prove,
};

fn parse(
    code: &str,
    public_inputs: JsonInputs,
    private_inputs: JsonInputs,
    debug: bool,
) -> Result<()> {
    // compile
    let (prover_index, verifier_index) = compile_and_prove(code)?;
    println!("successfuly compiled");

    // print ASM
    let asm = prover_index.asm(debug);
    println!("{asm}");

    // create proof
    let (proof, full_public_inputs, _public_output) =
        prover_index.prove(public_inputs, private_inputs, debug)?;
    println!("proof created");

    // verify proof
    verifier_index.verify(full_public_inputs, proof)?;
    println!("proof verified");

    Ok(())
}

#[derive(clap::Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Create a new noname package
    New(New),
    /// Create a new noname package in an existing directory
    Init,
    /// Build this package's and its dependencies' documentation
    Doc,
    /// Build the current package
    Build,
    /// Analyze the current package and report errors, but don't build object files
    Check,
    /// Add dependencies to a manifest file
    Add,
    /// Remove the target directory
    Clean,

    /// Run the main function and produce a proof
    Run,

    /// Verify a proof
    Verify,

    /// Compile, prove, and verify a noname program (for testing only)
    Test(Test),
}

#[derive(clap::Parser)]
struct New {}

#[derive(clap::Parser)]
struct Test {
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

    match cli.command {
        Commands::Test(test) => cmd_test(test),
        Commands::New(_) => todo!(),
        Commands::Init => todo!(),
        Commands::Doc => todo!(),
        Commands::Build => todo!(),
        Commands::Check => todo!(),
        Commands::Add => todo!(),
        Commands::Clean => todo!(),
        Commands::Run => todo!(),
        Commands::Verify => todo!(),
    }
}

fn cmd_test(test: Test) -> Result<()> {
    let code = std::fs::read_to_string(&test.path)
        .into_diagnostic()
        .wrap_err_with(|| "could not read file".to_string())?;

    let public_inputs = if let Some(s) = test.public_inputs {
        parse_inputs(&s)?
    } else {
        JsonInputs::default()
    };

    let private_inputs = if let Some(s) = test.private_inputs {
        parse_inputs(&s)?
    } else {
        JsonInputs::default()
    };

    parse(&code, public_inputs, private_inputs, test.debug).map_err(|e| e.with_source_code(code))
}

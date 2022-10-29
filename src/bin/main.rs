use std::{path::PathBuf, process};

use clap::Parser as _;
use miette::{IntoDiagnostic, Result, WrapErr};
use noname::cli::{
    cmd_build, cmd_check, cmd_init, cmd_new, cmd_test, CmdBuild, CmdCheck, CmdInit, CmdNew, CmdTest,
};

#[derive(clap::Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Create a new noname package
    New(CmdNew),
    /// Create a new noname package in an existing directory
    Init(CmdInit),
    /// Build this package's and its dependencies' documentation
    Doc,
    /// Build the current package
    Build(CmdBuild),
    /// Analyze the current package and report errors, but don't build object files
    Check(CmdCheck),
    /// Add dependencies to a manifest file
    Add,
    /// Remove the target directory
    Clean,

    /// Run the main function and produce a proof
    Run,

    /// Verify a proof
    Verify,

    /// Compile, prove, and verify a noname program (for testing only)
    Test(CmdTest),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Test(args) => cmd_test(args),
        Commands::New(args) => cmd_new(args),
        Commands::Init(args) => cmd_init(args),
        Commands::Doc => todo!(),
        Commands::Build(args) => cmd_build(args),
        Commands::Check(args) => cmd_check(args),
        Commands::Add => todo!(),
        Commands::Clean => todo!(),
        Commands::Run => todo!(),
        Commands::Verify => todo!(),
    }
}

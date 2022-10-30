use clap::Parser as _;
use miette::Result;
use noname::cli::{
    cmd_build, cmd_check, cmd_init, cmd_new, cmd_prove, cmd_verify, CmdBuild, CmdCheck, CmdInit,
    CmdNew, CmdProve, CmdVerify,
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

    // Build this package's and its dependencies' documentation. This command does not currently work
    //Doc,
    /// Build the current package
    Build(CmdBuild),

    /// Analyze the current package and report errors, but don't build object files
    Check(CmdCheck),

    // Add dependencies to a manifest file. This command does not currently work
    //Add,

    // Remove the target directory. This command does not currently work
    //Clean,
    /// Run the main function and produce a proof
    Run(CmdProve),

    /// An alias of the `--run` command.
    Prove(CmdProve),

    /// Verify a proof. This command does not currently work
    Verify(CmdVerify),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::New(args) => cmd_new(args),
        Commands::Init(args) => cmd_init(args),
        //        Commands::Doc => todo!(),
        Commands::Build(args) => cmd_build(args),
        Commands::Check(args) => cmd_check(args),
        //        Commands::Add => todo!(),
        //        Commands::Clean => todo!(),
        Commands::Run(args) => cmd_prove(args),
        Commands::Prove(args) => cmd_prove(args),
        Commands::Verify(args) => cmd_verify(args),
    }
}

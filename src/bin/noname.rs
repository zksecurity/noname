use clap::Parser as _;
use miette::Result;
#[cfg(feature = "cli")]
use noname::cli::{
    cmd_build, cmd_check, cmd_init, cmd_new, cmd_prove, cmd_run, cmd_test, cmd_verify, CmdBuild,
    CmdCheck, CmdInit, CmdNew, CmdProve, CmdRun, CmdTest, CmdVerify,
};

#[cfg(feature = "cli")]
#[derive(clap::Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "cli")]
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
    /// Generate circuit and witness
    Run(CmdRun),

    /// Run the main function and produce a proof
    Prove(CmdProve),

    /// Verify a proof. This command does not currently work
    Verify(CmdVerify),

    /// Tests a single file (as opposed to a package with a `Noname.toml` manifest file).
    /// This is intended for debugging, and should most likely not be used directly by users.
    /// This command will compile, attempt to create a proof, and verify it.
    Test(CmdTest),
}

#[cfg(feature = "cli")]
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
        Commands::Run(args) => cmd_run(args),
        Commands::Prove(args) => cmd_prove(args),
        Commands::Verify(args) => cmd_verify(args),

        Commands::Test(args) => cmd_test(args),
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    panic!("This binary requires the 'cli' feature to be enabled");
}

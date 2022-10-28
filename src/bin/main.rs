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
    New(CmdNew),
    /// Create a new noname package in an existing directory
    Init(CmdInit),
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
    Test(CmdTest),
}

#[derive(clap::Parser)]
struct CmdNew {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// is the package a library or a binary?
    #[clap(short, long)]
    lib: bool,
}

#[derive(clap::Parser)]
struct CmdInit {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// is the package a library or a binary?
    #[clap(short, long)]
    lib: bool,
}

#[derive(clap::Parser)]
struct CmdTest {
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
        Commands::Test(args) => cmd_test(args),
        Commands::New(args) => cmd_new(args),
        Commands::Init(args) => cmd_init(args),
        Commands::Doc => todo!(),
        Commands::Build => todo!(),
        Commands::Check => todo!(),
        Commands::Add => todo!(),
        Commands::Clean => todo!(),
        Commands::Run => todo!(),
        Commands::Verify => todo!(),
    }
}

fn cmd_new(args: CmdNew) -> Result<()> {
    let path = args.path;

    // for now, the package name is the same as the path
    let package_name = path
        .as_os_str()
        .to_str()
        .expect("couldn't parse the path as a string")
        .to_owned();

    if path.exists() {
        miette::bail!("path `{path}` already exists. Use `noname init` to create a new package in an existing directory");
    }

    std::fs::create_dir(&path)
        .into_diagnostic()
        .wrap_err("couldn't create directory at given path")?;

    mk(path, &package_name, args.lib)
}

fn cmd_init(args: CmdInit) -> Result<()> {
    let path = args.path;

    // for now, the package name is the same as the path
    let package_name = path
        .as_os_str()
        .to_str()
        .expect("couldn't parse the path as a string")
        .to_owned();

    if !path.exists() {
        miette::bail!("path `{path}` doesn't exists. Use `noname new` to create a new package in an non-existing directory");
    }

    mk(path, &package_name, args.lib)
}

fn mk(path: PathBuf, package_name: &str, is_lib: bool) -> Result<()> {
    let content = format!(
        r#"[package]
name = "{package_name}"
version = "0.1.0"
# see documentation at TODO for more information on how to edit this file

[dependencies]
"#
    );

    std::fs::write(path.join("Noname.toml"), content)
        .into_diagnostic()
        .wrap_err("cannot create Noname.toml file at given path")?;

    let src_path = path.join("src");

    std::fs::create_dir(&src_path)
        .into_diagnostic()
        .wrap_err("cannot create src directory at given path")?;

    if is_lib {
        let content = r#"fn add(xx: Field, yy: Field) -> Field {
    return xx + yy;
}
"#;

        std::fs::write(src_path.join("lib.no"), content)
            .into_diagnostic()
            .wrap_err("cannot create src/lib.no file at given path")?;
    } else {
        let content = r#"fn main(pub xx: Field, yy: Field) {
    let zz = yy + 1;
    assert_eq(zz, xx);
}
"#;
        std::fs::write(src_path.join("main.no"), content)
            .into_diagnostic()
            .wrap_err("cannot create src/main.no file at given path")?;
    }

    Ok(())
}

fn cmd_test(args: CmdTest) -> Result<()> {
    let code = std::fs::read_to_string(&args.path)
        .into_diagnostic()
        .wrap_err_with(|| "could not read file".to_string())?;

    let public_inputs = if let Some(s) = args.public_inputs {
        parse_inputs(&s)?
    } else {
        JsonInputs::default()
    };

    let private_inputs = if let Some(s) = args.private_inputs {
        parse_inputs(&s)?
    } else {
        JsonInputs::default()
    };

    parse(&code, public_inputs, private_inputs, args.debug).map_err(|e| e.with_source_code(code))
}

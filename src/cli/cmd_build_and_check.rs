use camino::Utf8PathBuf as PathBuf;
use miette::{Context, IntoDiagnostic, Result};

use crate::{
    circuit_writer::CircuitWriter,
    cli::packages::path_to_package,
    compiler::{compile_single, get_tast},
    inputs::{parse_inputs, JsonInputs},
    prover::{compile_to_indexes, ProverIndex, VerifierIndex},
    type_checker::{Dependencies, TypeChecker},
};

use super::packages::{
    get_deps_of_package, is_lib, validate_package_and_get_manifest, DependencyGraph, UserRepo,
};

const COMPILED_DIR: &str = "compiled";

#[derive(clap::Parser)]
pub struct CmdBuild {
    /// Path to the directory to create.
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,

    /// Prints an assembly-like encoding of the circuit.
    #[clap(long)]
    asm: bool,

    /// Prints a debug version of the assembly. To be used in conjunction with `--asm`.
    #[clap(long)]
    debug: bool,

    /// In case the path points to a binary,
    /// outputs the prover parameters to the given file.
    /// Defaults to `prover.nope`
    #[clap(long, value_parser)]
    prover_params: Option<PathBuf>,

    /// In case the path points to a binary,
    /// outputs the verifier parameters to the given file.
    /// Defaults to `verifier.nope`
    #[clap(long, value_parser)]
    verifier_params: Option<PathBuf>,
}

pub fn cmd_build(args: CmdBuild) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    let (prover_index, verifier_index) = build(&curr_dir, args.asm, args.debug)?;

    // create COMPILED_DIR
    let compiled_path = curr_dir.join(COMPILED_DIR);
    if compiled_path.is_dir() {
        miette::bail!("There's a filename called `{compiled_path}` which collides with noname. Please delete that file first." );
    }

    if args.prover_params.is_none() && args.verifier_params.is_none() && !compiled_path.exists() {
        std::fs::create_dir(&compiled_path)
            .into_diagnostic()
            .wrap_err(format!("could not create dir at `{compiled_path}`"))?;
    }

    // write prover
    let prover_params = args
        .prover_params
        .unwrap_or(compiled_path.join("prover.nope"));
    /*
    std::fs::write(&prover_params, rmp_serde::to_vec(&prover_index).unwrap())
        .into_diagnostic()
        .wrap_err(format!(
            "could not write prover params to `{prover_params}`"
        ))?;
        */

    // write verifier
    let verifier_params = args
        .verifier_params
        .unwrap_or(compiled_path.join("verifier.nope"));
    std::fs::write(
        &verifier_params,
        rmp_serde::to_vec(&verifier_index).unwrap(),
    )
    .into_diagnostic()
    .wrap_err(format!(
        "could not write prover params to `{prover_params}`"
    ))?;

    //
    Ok(())
}

#[derive(clap::Parser)]
pub struct CmdCheck {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,
}

pub fn cmd_check(args: CmdCheck) -> Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    // produce all TASTs and stop here
    produce_all_asts(&curr_dir)?;

    println!("all good!");
    Ok(())
}

fn produce_all_asts(path: &PathBuf) -> Result<(TypeChecker, Dependencies)> {
    // find manifest
    let manifest = validate_package_and_get_manifest(&path, false)?;

    // get all dependencies
    get_deps_of_package(&manifest);

    // produce dependency graph
    let is_lib = is_lib(&path);

    let this = if is_lib {
        Some(UserRepo::new(&manifest.package.name))
    } else {
        None
    };

    let dep_graph = DependencyGraph::new_from_manifest(this, &manifest)?;

    // produce artifacts for each dependency, starting from leaf dependencies
    let mut deps_tasts = Dependencies::default();
    for dep in dep_graph.from_leaves_to_roots() {
        let path = path_to_package(&dep);

        let lib_file = path.join("src").join("lib.no");
        let code = std::fs::read_to_string(&lib_file)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not read file `{path}`"))?;

        let lib_file_str = lib_file.to_string();
        let tast = get_tast(lib_file_str.clone(), code, &deps_tasts)?;
        deps_tasts.add_type_checker(dep, lib_file_str, tast);
    }

    // produce artifact for this one
    let src_dir = path.join("src");

    let lib_file = src_dir.join("lib.no");
    let main_file = src_dir.join("main.no");

    // assuming that we can't have both lib.no and main.no
    let file_path = if lib_file.exists() {
        lib_file
    } else {
        main_file
    };

    let code = std::fs::read_to_string(&file_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file `{file_path}`"))?;

    let tast = get_tast(file_path.to_string(), code, &deps_tasts)?;

    Ok((tast, deps_tasts))
}

pub fn build(
    curr_dir: &PathBuf,
    asm: bool,
    debug: bool,
) -> miette::Result<(ProverIndex, VerifierIndex)> {
    // produce all TASTs
    let (tast, deps_tasts) = produce_all_asts(curr_dir)?;

    // produce indexes
    let compiled_circuit = CircuitWriter::generate_circuit(tast, deps_tasts).into_diagnostic()?;

    if asm {
        println!("{}", compiled_circuit.asm(debug));
    }

    // TODO: cache artifacts

    // produce indexes
    compile_to_indexes(compiled_circuit)
}

#[derive(clap::Parser)]
pub struct CmdTest {
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

pub fn cmd_test(args: CmdTest) -> Result<()> {
    // read source code
    let code = std::fs::read_to_string(&args.path)
        .into_diagnostic()
        .wrap_err_with(|| {
            format!(
                "could not read file: `{}` (are you sure it exists?)",
                args.path
            )
        })?;

    // parse inputs
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

    // compile
    let compiled_circuit = compile_single(args.path.to_string(), code)?;
    let (prover_index, verifier_index) = compile_to_indexes(compiled_circuit)?;
    println!("successfuly compiled");

    // print ASM
    let asm = prover_index.asm(args.debug);
    println!("{asm}");

    // create proof
    let (proof, full_public_inputs, _public_output) =
        prover_index.prove(public_inputs, private_inputs, args.debug)?;
    println!("proof created");

    // verify proof
    verifier_index.verify(full_public_inputs, proof)?;
    println!("proof verified");

    Ok(())
}

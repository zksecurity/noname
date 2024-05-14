use camino::Utf8PathBuf as PathBuf;
use miette::{Context, IntoDiagnostic};

use crate::{
    backends::{
        kimchi::{
            prover::{ProverIndex, VerifierIndex},
            KimchiVesta,
        },
        r1cs::{snarkjs::SnarkjsExporter, R1CS},
        Backend, BackendField, BackendKind,
    },
    cli::packages::path_to_package,
    compiler::{compile, typecheck_next_file, Sources},
    inputs::{parse_inputs, JsonInputs},
    type_checker::TypeChecker,
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

    let (sources, prover_index, verifier_index) = build(&curr_dir, args.asm, args.debug)?;

    // create COMPILED_DIR
    let compiled_path = curr_dir.join(COMPILED_DIR);
    if compiled_path.exists() && !compiled_path.is_dir() {
        miette::bail!("There's a filename called `{}` which collides with noname. Please delete that file first.", compiled_path);
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

    println!("successfully built");

    //
    Ok(())
}

#[derive(clap::Parser)]
pub struct CmdCheck {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,
}

pub fn cmd_check(args: CmdCheck) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    // produce all TASTs and stop here
    produce_all_asts::<KimchiVesta>(&curr_dir)?;

    println!("all good!");
    Ok(())
}

fn produce_all_asts<B: Backend>(path: &PathBuf) -> miette::Result<(Sources, TypeChecker<B>)> {
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
    let mut sources = Sources::new();
    let mut node_id = 0;

    let mut tast = TypeChecker::new();

    for dep in dep_graph.from_leaves_to_roots() {
        let path = path_to_package(&dep);

        let lib_file = path.join("src").join("lib.no");
        let code = std::fs::read_to_string(&lib_file)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not read file `{path}`"))?;

        let lib_file_str = lib_file.to_string();
        node_id = typecheck_next_file(
            &mut tast,
            Some(dep),
            &mut sources,
            lib_file_str.clone(),
            code,
            node_id,
        )?;
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

    let _node_id = typecheck_next_file(
        &mut tast,
        None,
        &mut sources,
        file_path.to_string(),
        code,
        node_id,
    )?;

    Ok((sources, tast))
}

pub fn build(
    curr_dir: &PathBuf,
    asm: bool,
    debug: bool,
) -> miette::Result<(Sources, ProverIndex, VerifierIndex)> {
    // produce all TASTs
    let (sources, tast) = produce_all_asts(curr_dir)?;

    // produce indexes
    let double_generic_gate_optimization = false;

    let kimchi_vesta = KimchiVesta::new(double_generic_gate_optimization);
    let compiled_circuit = compile(&sources, tast, kimchi_vesta)?;

    if asm {
        println!("{}", compiled_circuit.asm(&sources, debug));
    }

    // TODO: cache artifacts

    // produce indexes
    let (prover_index, verifier_index) = compiled_circuit.compile_to_indexes()?;

    Ok((sources, prover_index, verifier_index))
}

#[derive(clap::Parser)]
pub struct CmdTest {
    /// path to the .no file
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// Backend to use for running the noname file.
    /// supported backends:
    /// - `snarkjs-r1cs`
    #[clap(short, long, value_parser)]
    backend: String,

    /// public inputs in a JSON format using decimal values (e.g. {"a": "1", "b": "2"})
    #[clap(long)]
    public_inputs: Option<String>,

    /// private inputs in a JSON format using decimal values (e.g. {"a": "1", "b": "2"})
    #[clap(long)]
    private_inputs: Option<String>,

    /// prints debug information (defaults to false)
    #[clap(short, long)]
    debug: bool,

    /// enable the double generic gate optimization of kimchi (by default noname uses that optimization)
    #[clap(long)]
    double: bool,
}

pub fn cmd_test(args: CmdTest) -> miette::Result<()> {
    let backend = args.backend;

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

    let backend_kind = match backend.as_str() {
        "kimchi-vesta" => BackendKind::new_kimchi_vesta(false),
        "r1cs-bls12-381" => BackendKind::new_r1cs_bls12_381(),
        "r1cs-bn254" => BackendKind::new_r1cs_bn254(),
        _ => miette::bail!("unknown backend: `{}`", backend),
    };

    match backend_kind {
        BackendKind::KimchiVesta(_) => {
            let (tast, sources) = typecheck_file(&args.path)?;
            let kimchi_vesta = KimchiVesta::new(args.double);
            let compiled_circuit = compile(&sources, tast, kimchi_vesta)?;

            let (prover_index, verifier_index) = compiled_circuit.compile_to_indexes()?;
            println!("successfully compiled");

            // print ASM
            let asm = prover_index.asm(&sources, args.debug);
            println!("{asm}");

            // create proof
            let (proof, full_public_inputs, _public_output) =
                prover_index.prove(&sources, public_inputs, private_inputs, args.debug)?;
            println!("proof created");

            // verify proof
            verifier_index.verify(full_public_inputs, proof)?;
            println!("proof verified");
        }
        BackendKind::R1csBls12_381(r1cs) => {
            test_r1cs_backend(r1cs, &args.path, public_inputs, private_inputs, args.debug)?;
        }
        BackendKind::R1csBn254(r1cs) => {
            test_r1cs_backend(r1cs, &args.path, public_inputs, private_inputs, args.debug)?;
        }
    }

    Ok(())
}

#[derive(clap::Parser)]
pub struct CmdRun {
    /// noname file to run
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,

    /// Backend to use for running the noname file.
    #[clap(
        short,
        long,
        value_parser,
        help = "Supported backends: `kimchi-vesta`, `r1cs-bls12-381`, `r1cs-bn254"
    )]
    backend: Option<String>,

    /// JSON encoding of the public inputs. For example: `--public-inputs {"a": "1", "b": ["2", "3"]}`.
    #[clap(long, value_parser, default_value = "{}")]
    public_inputs: Option<String>,

    /// JSON encoding of the private inputs. Similar to `--public-inputs` but for private inputs.
    #[clap(long, value_parser, default_value = "{}")]
    private_inputs: Option<String>,
}

pub fn cmd_run(args: CmdRun) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    let backend = args.backend.unwrap();

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

    let backend_kind = match backend.as_str() {
        "kimchi-vesta" => BackendKind::new_kimchi_vesta(false),
        "r1cs-bls12-381" => BackendKind::new_r1cs_bls12_381(),
        "r1cs-bn254" => BackendKind::new_r1cs_bn254(),
        _ => miette::bail!("unknown backend: `{}`", backend),
    };

    match backend_kind {
        BackendKind::KimchiVesta(_) => {
            unimplemented!("kimchi-vesta backend is not yet supported for this command")
        }
        BackendKind::R1csBls12_381(r1cs) => {
            run_r1cs_backend(r1cs, &curr_dir, public_inputs, private_inputs)?
        }
        BackendKind::R1csBn254(r1cs) => {
            run_r1cs_backend(r1cs, &curr_dir, public_inputs, private_inputs)?
        }
    }

    Ok(())
}

fn run_r1cs_backend<F>(
    r1cs: R1CS<F>,
    curr_dir: &PathBuf,
    public_inputs: JsonInputs,
    private_inputs: JsonInputs,
) -> miette::Result<()>
where
    F: BackendField,
{
    // Assuming `curr_dir`, `public_inputs`, and `private_inputs` are available in the scope
    let (sources, tast) = produce_all_asts(curr_dir)?;
    println!("running noname file");

    let compiled_circuit = compile(&sources, tast, r1cs)?;

    let generated_witness = compiled_circuit.generate_witness(public_inputs, private_inputs)?;

    let snarkjs_exporter = SnarkjsExporter::new(compiled_circuit.circuit.backend);

    snarkjs_exporter.gen_r1cs_file(&curr_dir.join("output.r1cs").into_string());

    snarkjs_exporter.gen_wtns_file(
        &curr_dir.join("output.wtns").into_string(),
        generated_witness,
    );

    Ok(())
}

fn test_r1cs_backend<F: BackendField>(
    r1cs: R1CS<F>,
    path: &PathBuf,
    public_inputs: JsonInputs,
    private_inputs: JsonInputs,
    debug: bool,
) -> miette::Result<()>
where
    F: BackendField,
{
    let (tast, sources) = typecheck_file(path)?;

    let compiled_circuit = compile(&sources, tast, r1cs)?;
    println!("successfully compiled");

    compiled_circuit.generate_witness(public_inputs, private_inputs)?;

    let asm = compiled_circuit.asm(&sources, debug);

    println!("{}", asm);

    Ok(())
}

fn typecheck_file<B: Backend>(path: &PathBuf) -> miette::Result<(TypeChecker<B>, Sources)> {
    let code = std::fs::read_to_string(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file: `{}` (are you sure it exists?)", path))?;

    let mut sources = Sources::new();
    let mut tast = TypeChecker::<B>::new();
    let _node_id = typecheck_next_file(&mut tast, None, &mut sources, path.to_string(), code, 0)?;

    Ok((tast, sources))
}

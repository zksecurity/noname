use camino::Utf8PathBuf as PathBuf;
use miette::{Context, IntoDiagnostic};

use crate::inputs::parse_inputs;

use super::cmd_build_and_check::build;

#[derive(clap::Parser)]
pub struct CmdProve {
    /// Path to the directory to create.
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,

    /// Prints the witness.
    #[clap(long)]
    debug: bool,

    /// Path to the resulting proof. Defaults to `proof.nope`.
    #[clap(long, value_parser)]
    proof_path: Option<PathBuf>,

    /// JSON encoding of the public inputs. For example: `--public-inputs {"a": "1", "b": ["2", "3"]}`.
    #[clap(long, value_parser, default_value = "{}")]
    public_inputs: String,

    /// JSON encoding of the private inputs. Similar to `--public-inputs` but for private inputs.
    #[clap(long, value_parser, default_value = "{}")]
    private_inputs: String,
}

pub fn cmd_prove(args: CmdProve) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    let (sources, prover_index, verifier_index) = build(&curr_dir, false, args.debug)?;

    // parse inputs
    let public_inputs = parse_inputs(&args.public_inputs).unwrap();
    let private_inputs = parse_inputs(&args.private_inputs).unwrap();

    // create proof
    let (proof, full_public_inputs, public_output) =
        prover_index.prove(&sources, public_inputs, private_inputs, args.debug)?;

    // verify proof
    if args.debug {
        verifier_index.verify(full_public_inputs, proof.clone())?;
    }

    // serialize proof
    let proof_path = args
        .proof_path
        .unwrap_or_else(|| curr_dir.join("proof.nope"));
    std::fs::write(&proof_path, rmp_serde::to_vec(&proof).unwrap())
        .into_diagnostic()
        .wrap_err(format!("could not write the proof to `{proof_path}`"))?;

    // notification
    if public_output.is_empty() {
        println!(
            "proof created at path `{proof_path}`. You can use `noname --verify` to verify it. Note that you will need to pass the same JSON-encoded public inputs as you did when creating the proof. (If you didn't use the `--public-inputs` flag, then you don't need to pass any public inputs.)",
        );
    } else {
        println!("proof created at path `{proof_path}`. Since running the proof produced a  public output `{public_output:?}`, you will need to also pass the expected public output to the verifier (who can run `noname verify --public-output '{public_output:?}'`).");
    }

    //
    Ok(())
}

#[derive(clap::Parser)]
pub struct CmdVerify {
    /// Path to the directory to create.
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,

    /// Path to the proof to verify. Defaults to `proof.nope`.
    #[clap(short, long, value_parser)]
    proof_path: Option<PathBuf>,

    /// JSON encoding of the public inputs. For example: `--public-inputs {"a": "1", "b": ["2", "3"]}`.
    #[clap(short, long, value_parser)]
    public_inputs: String,

    /// An optional expected public output, in JSON format.
    #[clap(short, long, value_parser)]
    public_output: Option<String>,
}

pub fn cmd_verify(args: CmdVerify) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap());

    let (_sources, _prover_index, _verifier_index) = build(&curr_dir, false, false)?;

    // parse inputs
    let _public_inputs = parse_inputs(&args.public_inputs).unwrap();

    if let Some(public_output) = &args.public_output {
        let _public_output = parse_inputs(public_output).unwrap();

        // TODO: add it to the public input
        todo!();
    }

    // get proof
    let proof_path = args
        .proof_path
        .unwrap_or_else(|| curr_dir.join("proof.nope"));

    if !proof_path.exists() {
        miette::bail!("proof does not exist at path `{proof_path}`. Perhaps pass the correct path via the `--proof-path` flag?");
    }

    rmp_serde::from_read(std::fs::File::open(&proof_path).unwrap())
        .into_diagnostic()
        .wrap_err(format!(
            "could not deserialize the given proof at `{proof_path}`"
        ))?;

    // verify proof
    unimplemented!();
    /*
        verifier_index
            .verify(full_public_inputs, proof)
            .into_diagnostic()
            .wrap_err("Failed to verify proof")?;
    */
    //
}

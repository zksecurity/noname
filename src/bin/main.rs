use std::{collections::HashMap, path::PathBuf};

use ark_ff::One;
use clap::Parser;
use miette::{IntoDiagnostic, Result, WrapErr};
use my_programming_language::{
    ast::{Compiler, Gate},
    constants::{Field, COLUMNS},
    lexer::Token,
    parser::AST,
};

fn parse(name: impl std::fmt::Display, code: &str) -> Result<()> {
    let tokens = Token::parse(code)?;
    let ast = AST::parse(tokens)?;
    let (circuit, compiler) = Compiler::analyze_and_compile(ast, code)?;

    println!("compiled {name} to {} gates", compiler.num_gates());

    println!("\n==== circuit ====\n");

    println!("{circuit}");

    println!("\n========\n");

    // generate witness
    let mut args = HashMap::new();
    args.insert("public_input", Field::one());
    args.insert("private_input", Field::one());
    let witness = compiler.generate_witness(args)?;

    // create proof

    let public_inputs = vec![Field::one()];

    //witness.scramble();
    prove_and_verify(
        &compiler.compiled_gates(),
        witness.to_kimchi_witness(),
        public_inputs,
    );

    Ok(())
}

fn prove_and_verify(gates: &[Gate], witness: [Vec<Field>; COLUMNS], public: Vec<Field>) {
    // convert gates
    let gates: Vec<_> = gates
        .into_iter()
        .enumerate()
        .map(|(row, gate)| gate.to_kimchi_gate(row))
        .collect();

    // create constraint system
    let fp_sponge_params = kimchi::oracle::pasta::fp_kimchi::params();

    let cs = kimchi::circuits::constraints::ConstraintSystem::create(gates, fp_sponge_params)
        .public(public.len())
        .build()
        .unwrap();

    // create SRS (for vesta, as the circuit is in Fp)
    type Curve = kimchi::mina_curves::pasta::vesta::Affine;
    type OtherCurve = kimchi::mina_curves::pasta::pallas::Affine;

    let mut srs = kimchi::commitment_dlog::srs::SRS::<Curve>::create(cs.domain.d1.size as usize);
    srs.add_lagrange_basis(cs.domain.d1);
    let srs = std::sync::Arc::new(srs);

    println!("using an SRS of size {}", srs.g.len());

    // create index
    let fq_sponge_params = kimchi::oracle::pasta::fq_kimchi::params();
    let (endo_q, _endo_r) = kimchi::commitment_dlog::srs::endos::<OtherCurve>();

    let prover_index =
        kimchi::prover_index::ProverIndex::<Curve>::create(cs, fq_sponge_params, endo_q, srs);
    let verifier_index = prover_index.verifier_index();

    // verify the witness
    prover_index.cs.verify(&witness, &public).unwrap();

    // create proof
    use kimchi::groupmap::GroupMap;
    let group_map = <Curve as kimchi::commitment_dlog::commitment::CommitmentCurve>::Map::setup();

    type SpongeParams = kimchi::oracle::constants::PlonkSpongeConstantsKimchi;
    type BaseSponge = kimchi::oracle::sponge::DefaultFqSponge<
        kimchi::mina_curves::pasta::vesta::VestaParameters,
        SpongeParams,
    >;
    type ScalarSponge = kimchi::oracle::sponge::DefaultFrSponge<Field, SpongeParams>;

    let proof = kimchi::proof::ProverProof::create::<BaseSponge, ScalarSponge>(
        &group_map,
        witness,
        &[],
        &prover_index,
    )
    .unwrap();

    // verify the proof
    kimchi::verifier::verify::<Curve, BaseSponge, ScalarSponge>(
        &group_map,
        &verifier_index,
        &proof,
    )
    .unwrap();
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser)]
    path: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let code = std::fs::read_to_string(&cli.path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file"))?;

    parse(cli.path.display(), &code).map_err(|e| e.with_source_code(code))?;

    println!("successfuly compiled");

    Ok(())
}

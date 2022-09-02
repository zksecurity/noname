//! This module contains the prover.

use crate::{
    ast::Compiler,
    error::{Error, Result},
    field::Field,
    inputs::Inputs,
    lexer::Token,
    parser::AST,
};

use clap::once_cell::sync::Lazy;
use kimchi::{
    commitment_dlog::commitment::CommitmentCurve, groupmap::GroupMap, proof::ProverProof,
};

//
// aliases
//

type Curve = kimchi::mina_curves::pasta::vesta::Affine;
type OtherCurve = kimchi::mina_curves::pasta::pallas::Affine;
type SpongeParams = kimchi::oracle::constants::PlonkSpongeConstantsKimchi;
type BaseSponge = kimchi::oracle::sponge::DefaultFqSponge<
    kimchi::mina_curves::pasta::vesta::VestaParameters,
    SpongeParams,
>;
type ScalarSponge = kimchi::oracle::sponge::DefaultFrSponge<Field, SpongeParams>;

//
// Lazy static
//

static GROUP_MAP: Lazy<<Curve as CommitmentCurve>::Map> =
    Lazy::new(<Curve as CommitmentCurve>::Map::setup);

//
// Data Structures
//

pub struct ProverIndex {
    index: kimchi::prover_index::ProverIndex<Curve>,
    compiler: Compiler,
}

pub struct VerifierIndex {
    index: kimchi::verifier_index::VerifierIndex<Curve>,
}

//
// Setup
//

pub fn compile(code: &str, debug: bool) -> Result<(String, ProverIndex, VerifierIndex)> {
    // lexer
    let tokens = Token::parse(code)?;

    // AST
    let ast = AST::parse(tokens)?;

    // type checker + compiler
    let (asm, compiler) = Compiler::analyze_and_compile(ast, code, debug)?;

    // convert gates to kimchi gates
    let gates: Vec<_> = compiler
        .compiled_gates()
        .iter()
        .enumerate()
        .map(|(row, gate)| gate.to_kimchi_gate(row))
        .collect();

    // create constraint system
    let fp_sponge_params = kimchi::oracle::pasta::fp_kimchi::params();

    let cs = kimchi::circuits::constraints::ConstraintSystem::create(gates, fp_sponge_params)
        .public(compiler.public_input_size)
        .build()
        .map_err(|e| Error {
            kind: e.into(),
            span: (0, 0),
        })?;

    // create SRS (for vesta, as the circuit is in Fp)

    let mut srs = kimchi::commitment_dlog::srs::SRS::<Curve>::create(cs.domain.d1.size as usize);
    srs.add_lagrange_basis(cs.domain.d1);
    let srs = std::sync::Arc::new(srs);

    println!("using an SRS of size {}", srs.g.len());

    // create indexes
    let fq_sponge_params = kimchi::oracle::pasta::fq_kimchi::params();
    let (endo_q, _endo_r) = kimchi::commitment_dlog::srs::endos::<OtherCurve>();

    let prover_index =
        kimchi::prover_index::ProverIndex::<Curve>::create(cs, fq_sponge_params, endo_q, srs);
    let verifier_index = prover_index.verifier_index();

    // wrap
    let prover_index = {
        ProverIndex {
            index: prover_index,
            compiler,
        }
    };
    let verifier_index = VerifierIndex {
        index: verifier_index,
    };

    // return asm + indexes
    Ok((asm, prover_index, verifier_index))
}

//
// Proving
//

impl ProverIndex {
    /// returns a proof and a public output
    pub fn prove(
        &self,
        public_inputs: Inputs,
        private_inputs: Inputs,
        debug: bool,
    ) -> Result<(ProverProof<Curve>, Vec<Field>, Vec<Field>)> {
        // generate the witness
        let (witness, full_public_inputs, public_output) = self
            .compiler
            .generate_witness(public_inputs, private_inputs)?;

        if debug {
            println!("# witness\n");
            witness.debug();
        }

        // convert to kimchi format
        let witness = witness.to_kimchi_witness();

        // verify the witness
        if debug {
            self.index.cs.verify(&witness, &full_public_inputs).unwrap();
        }

        // create proof
        let proof =
            ProverProof::create::<BaseSponge, ScalarSponge>(&GROUP_MAP, witness, &[], &self.index)
                .map_err(|e| Error {
                    kind: e.into(),
                    span: (0, 0),
                });

        // return proof + public output
        proof.map(|proof| (proof, full_public_inputs, public_output))
    }
}

//
// Verifying
//

impl VerifierIndex {
    pub fn verify(&self, full_public_inputs: Vec<Field>, proof: ProverProof<Curve>) -> Result<()> {
        // pass the public input in the proof
        let mut proof = proof;
        proof.public = full_public_inputs;

        // verify the proof
        kimchi::verifier::verify::<Curve, BaseSponge, ScalarSponge>(&GROUP_MAP, &self.index, &proof)
            .map_err(|e| Error {
                kind: e.into(),
                span: (0, 0),
            })
    }
}

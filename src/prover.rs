//! This module contains the prover.

use std::iter::once;

use crate::{
    circuit_writer::Wiring,
    compiler,
    constants::{Field, Span},
    error::{Error, Result},
    inputs::JsonInputs,
    witness::CompiledCircuit,
};

use clap::once_cell::sync::Lazy;
use itertools::chain;
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
    circuit: CompiledCircuit,
}

pub struct VerifierIndex {
    index: kimchi::verifier_index::VerifierIndex<Curve>,
}

//
// Setup
//

pub fn compile_and_prove(code: &str) -> Result<(ProverIndex, VerifierIndex)> {
    let circuit = compiler::compile(code)?;

    // convert gates to kimchi gates
    let mut gates: Vec<_> = circuit
        .compiled_gates()
        .iter()
        .enumerate()
        .map(|(row, gate)| gate.to_kimchi_gate(row))
        .collect();

    // wiring
    for wiring in circuit.wiring.values() {
        if let Wiring::Wired(cells_and_spans) = wiring {
            // all the wired cells form a cycle, remember!
            let mut wired_cells = cells_and_spans.iter().map(|(cell, _)| cell).copied();
            assert!(wired_cells.len() > 1);

            let first_cell = wired_cells.next().unwrap(); // for the cycle
            let mut prev_cell = first_cell;

            for cell in chain![wired_cells, once(first_cell)] {
                gates[cell.row].wires[cell.col] = kimchi::circuits::wires::Wire {
                    row: prev_cell.row,
                    col: prev_cell.col,
                };
                prev_cell = cell;
            }
        }
    }

    // create constraint system
    let fp_sponge_params = kimchi::oracle::pasta::fp_kimchi::params();

    let cs = kimchi::circuits::constraints::ConstraintSystem::create(gates, fp_sponge_params)
        .public(circuit.public_input_size)
        .build()
        .map_err(|e| Error::new(e.into(), Span(0, 0)))?;

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
            circuit,
        }
    };
    let verifier_index = VerifierIndex {
        index: verifier_index,
    };

    // return asm + indexes
    Ok((prover_index, verifier_index))
}

//
// Proving
//

impl ProverIndex {
    pub fn asm(&self, debug: bool) -> String {
        self.circuit.asm(debug)
    }

    /// returns a proof and a public output
    pub fn prove(
        &self,
        public_inputs: JsonInputs,
        private_inputs: JsonInputs,
        debug: bool,
    ) -> Result<(ProverProof<Curve>, Vec<Field>, Vec<Field>)> {
        // generate the witness
        let (witness, full_public_inputs, public_output) = self
            .circuit
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
                .map_err(|e| Error::new(e.into(), Span(0, 0)));

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
            .map_err(|e| Error::new(e.into(), Span(0, 0)))
    }
}

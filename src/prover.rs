//! This module contains the prover.

use std::iter::once;

use crate::{
    circuit_writer::Wiring, constants::Field, inputs::JsonInputs, witness::CompiledCircuit,
};

use itertools::chain;
use kimchi::{
    commitment_dlog::commitment::CommitmentCurve, groupmap::GroupMap, proof::ProverProof,
};
use miette::{Context, IntoDiagnostic};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

//
// aliases
//

type Curve = kimchi::mina_curves::pasta::Vesta;
type OtherCurve = kimchi::mina_curves::pasta::Pallas;
type SpongeParams = kimchi::oracle::constants::PlonkSpongeConstantsKimchi;
type BaseSponge = kimchi::oracle::sponge::DefaultFqSponge<
    kimchi::mina_curves::pasta::VestaParameters,
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

//#[derive(Serialize, Deserialize)]
pub struct ProverIndex {
    index: kimchi::prover_index::ProverIndex<Curve>,
    compiled_circuit: CompiledCircuit,
}

#[derive(Serialize, Deserialize)]
pub struct VerifierIndex {
    index: kimchi::verifier_index::VerifierIndex<Curve>,
}

//
// Setup
//

pub fn compile_to_indexes(
    compiled_circuit: CompiledCircuit,
) -> miette::Result<(ProverIndex, VerifierIndex)> {
    // convert gates to kimchi gates
    let mut gates: Vec<_> = compiled_circuit
        .compiled_gates()
        .iter()
        .enumerate()
        .map(|(row, gate)| gate.to_kimchi_gate(row))
        .collect();

    // wiring
    for wiring in compiled_circuit.circuit.wiring.values() {
        if let Wiring::Wired(annotated_cells) = wiring {
            // all the wired cells form a cycle, remember!
            let mut wired_cells = annotated_cells
                .iter()
                .map(|annotated_cell| annotated_cell.cell);
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
    let cs = kimchi::circuits::constraints::ConstraintSystem::create(gates)
        .public(compiled_circuit.circuit.public_input_size)
        .build()
        .into_diagnostic()
        .wrap_err("kimchi: could not create a constraint system with the given circuit and public input size")?;

    // create SRS (for vesta, as the circuit is in Fp)
    let mut srs = kimchi::commitment_dlog::srs::SRS::<Curve>::create(cs.domain.d1.size as usize);
    srs.add_lagrange_basis(cs.domain.d1);
    let srs = std::sync::Arc::new(srs);

    println!("using an SRS of size {}", srs.g.len());

    // create indexes
    let (endo_q, _endo_r) = kimchi::commitment_dlog::srs::endos::<OtherCurve>();

    let prover_index = kimchi::prover_index::ProverIndex::<Curve>::create(cs, endo_q, srs);
    let verifier_index = prover_index.verifier_index();

    // wrap
    let prover_index = {
        ProverIndex {
            index: prover_index,
            compiled_circuit,
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
        self.compiled_circuit.asm(debug)
    }

    pub fn len(&self) -> usize {
        self.compiled_circuit.compiled_gates().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a proof and a public output
    pub fn prove(
        &self,
        public_inputs: JsonInputs,
        private_inputs: JsonInputs,
        debug: bool,
    ) -> miette::Result<(ProverProof<Curve>, Vec<Field>, Vec<Field>)> {
        // generate the witness
        let (witness, full_public_inputs, public_output) = self
            .compiled_circuit
            .generate_witness(public_inputs, private_inputs)?;

        if debug {
            println!("# witness\n");
            witness.debug();
        }

        // convert to kimchi format
        let witness = witness.to_kimchi_witness();

        // verify the witness
        if debug {
            self.index
                .cs
                .verify::<Curve>(&witness, &full_public_inputs)
                .unwrap();
        }

        // create proof
        let proof =
            ProverProof::create::<BaseSponge, ScalarSponge>(&GROUP_MAP, witness, &[], &self.index)
                .into_diagnostic()
                .wrap_err("kimchi: could not create a proof with the given inputs")?;

        // return proof + public output
        Ok((proof, full_public_inputs, public_output))
    }
}

//
// Verifying
//

impl VerifierIndex {
    pub fn verify(
        &self,
        full_public_inputs: Vec<Field>,
        proof: ProverProof<Curve>,
    ) -> miette::Result<()> {
        // pass the public input in the proof
        let mut proof = proof;
        proof.public = full_public_inputs;

        // verify the proof
        kimchi::verifier::verify::<Curve, BaseSponge, ScalarSponge>(&GROUP_MAP, &self.index, &proof)
            .into_diagnostic()
            .wrap_err("kimchi: failed to verify the proof")
    }
}

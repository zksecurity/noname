//! This module contains the prover.

use std::iter::once;

use crate::{
    backends::{
        kimchi::{Kimchi, KimchiField},
        Backend,
    },
    circuit_writer::Wiring,
    compiler::{generate_witness, Sources},
    inputs::JsonInputs,
    witness::CompiledCircuit,
};

use itertools::chain;
use kimchi::mina_curves::pasta::{Vesta, VestaParameters};
use kimchi::mina_poseidon::constants::PlonkSpongeConstantsKimchi;
use kimchi::mina_poseidon::sponge::{DefaultFqSponge, DefaultFrSponge};
use kimchi::poly_commitment::commitment::CommitmentCurve;
use kimchi::poly_commitment::evaluation_proof::OpeningProof;
use kimchi::proof::ProverProof;
use kimchi::{
    circuits::constraints::ConstraintSystem, groupmap::GroupMap, mina_curves::pasta::Pallas,
    poly_commitment::srs::SRS,
};

use miette::{Context, IntoDiagnostic};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

//
// aliases
//

type Curve = Vesta;
type OtherCurve = Pallas;
type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams>;
type ScalarSponge = DefaultFrSponge<kimchi::mina_curves::pasta::Fp, SpongeParams>;

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
    index: kimchi::prover_index::ProverIndex<Curve, OpeningProof<Curve>>,
    compiled_circuit: CompiledCircuit<Kimchi>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifierIndex {
    index: kimchi::verifier_index::VerifierIndex<Curve, OpeningProof<Curve>>,
}

//
// Setup
//

pub fn compile_to_indexes(
    compiled_circuit: CompiledCircuit<Kimchi>,
) -> miette::Result<(ProverIndex, VerifierIndex)> {
    // convert gates to kimchi gates
    let mut gates: Vec<_> = compiled_circuit
        .circuit
        .backend
        .gates()
        .iter()
        .enumerate()
        .map(|(row, gate)| gate.to_kimchi_gate(row))
        .collect();

    // wiring
    for wiring in compiled_circuit.circuit.backend.wiring.values() {
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
    let cs = ConstraintSystem::create(gates)
        .public(compiled_circuit.circuit.public_input_size)
        .build()
        .into_diagnostic()
        .wrap_err("kimchi: could not create a constraint system with the given circuit and public input size")?;

    // create SRS (for vesta, as the circuit is in Fp)
    let mut srs = SRS::<Curve>::create(cs.domain.d1.size as usize);
    srs.add_lagrange_basis(cs.domain.d1);
    let srs = std::sync::Arc::new(srs);

    println!("using an SRS of size {}", srs.g.len());

    // create indexes
    let (endo_q, _endo_r) = kimchi::poly_commitment::srs::endos::<OtherCurve>();

    let prover_index =
        kimchi::prover_index::ProverIndex::<Curve, OpeningProof<Curve>>::create(cs, endo_q, srs);
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
    pub fn asm(&self, sources: &Sources, debug: bool) -> String {
        self.compiled_circuit.asm(sources, debug)
    }

    pub fn len(&self) -> usize {
        self.compiled_circuit.circuit.backend.gates().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a proof and a public output
    pub fn prove(
        &self,
        sources: &Sources,
        public_inputs: JsonInputs,
        private_inputs: JsonInputs,
        debug: bool,
    ) -> miette::Result<(
        ProverProof<Curve, OpeningProof<Curve>>,
        Vec<KimchiField>,
        Vec<KimchiField>,
    )> {
        // generate the witness
        let (witness, full_public_inputs, public_output) = generate_witness(
            &self.compiled_circuit,
            sources,
            public_inputs,
            private_inputs,
        )?;

        if debug {
            println!("# witness\n");
            witness.debug();
        }

        // convert to kimchi format
        let witness = witness.to_kimchi_witness();

        // verify the witness
        if debug {
            self.index.verify(&witness, &full_public_inputs).unwrap();
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
        full_public_inputs: Vec<KimchiField>,
        proof: ProverProof<Curve, OpeningProof<Curve>>,
    ) -> miette::Result<()> {
        // verify the proof
        kimchi::verifier::verify::<Curve, BaseSponge, ScalarSponge, OpeningProof<Curve>>(
            &GROUP_MAP,
            &self.index,
            &proof,
            &full_public_inputs,
        )
        .into_diagnostic()
        .wrap_err("kimchi: failed to verify the proof")
    }
}

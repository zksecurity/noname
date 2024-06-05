//! This module contains the prover.

use std::iter::once;

use crate::{
    backends::kimchi::{KimchiVesta, VestaField},
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
    compiled_circuit: CompiledCircuit<KimchiVesta>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifierIndex {
    index: kimchi::verifier_index::VerifierIndex<Curve, OpeningProof<Curve>>,
}

//
// Setup
//

#[allow(clippy::type_complexity)]
impl KimchiVesta {
    pub fn compile_to_indexes(
        &self,
    ) -> miette::Result<(
        kimchi::prover_index::ProverIndex<Curve, OpeningProof<Curve>>,
        kimchi::verifier_index::VerifierIndex<Curve, OpeningProof<Curve>>,
    )> {
        // convert gates to kimchi gates
        let mut gates: Vec<_> = self
            .gates
            .iter()
            .enumerate()
            .map(|(row, gate)| gate.to_kimchi_gate(row))
            .collect();

        // wiring
        for wiring in self.wiring.values() {
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
            .public(self.public_input_size)
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

        let prover_index = kimchi::prover_index::ProverIndex::<Curve, OpeningProof<Curve>>::create(
            cs, endo_q, srs,
        );
        let verifier_index = prover_index.verifier_index();

        Ok((prover_index, verifier_index))
    }
}

impl CompiledCircuit<KimchiVesta> {
    pub fn compile_to_indexes(self) -> miette::Result<(ProverIndex, VerifierIndex)> {
        let (prover_index, verifier_index) = self.circuit.backend.compile_to_indexes()?;
        // wrap
        let prover_index = {
            ProverIndex {
                index: prover_index,
                compiled_circuit: self,
            }
        };
        let verifier_index = VerifierIndex {
            index: verifier_index,
        };

        // return asm + indexes
        Ok((prover_index, verifier_index))
    }
}

//
// Proving
//

impl ProverIndex {
    pub fn asm(&self, sources: &Sources, debug: bool) -> String {
        self.compiled_circuit.asm(sources, debug)
    }

    pub fn len(&self) -> usize {
        self.compiled_circuit.circuit.backend.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a proof and a public output
    #[allow(clippy::type_complexity)]
    pub fn prove(
        &self,
        sources: &Sources,
        public_inputs: JsonInputs,
        private_inputs: JsonInputs,
        debug: bool,
    ) -> miette::Result<(
        ProverProof<Curve, OpeningProof<Curve>>,
        Vec<VestaField>,
        Vec<VestaField>,
    )> {
        // generate the witness
        let generated_witness = generate_witness(
            &self.compiled_circuit,
            sources,
            public_inputs,
            private_inputs,
        )?;

        if debug {
            println!("# witness\n");
            generated_witness.all_witness.debug();
        }

        // convert to kimchi format
        let witness = generated_witness.all_witness.to_kimchi_witness();

        // verify the witness
        if debug {
            self.index
                .verify(&witness, &generated_witness.full_public_inputs)
                .unwrap();
        }

        // create proof
        let proof =
            ProverProof::create::<BaseSponge, ScalarSponge>(&GROUP_MAP, witness, &[], &self.index)
                .into_diagnostic()
                .wrap_err("kimchi: could not create a proof with the given inputs")?;

        // return proof + public output
        Ok((
            proof,
            generated_witness.full_public_inputs,
            generated_witness.public_outputs,
        ))
    }
}

//
// Verifying
//

impl VerifierIndex {
    pub fn verify(
        &self,
        full_public_inputs: Vec<VestaField>,
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

#[cfg(test)]
mod tests {
    use kimchi::circuits::constraints::GateError;

    use crate::{
        backends::kimchi::{KimchiVesta, VestaField},
        compiler::{compile, generate_witness, typecheck_next_file, Sources},
        inputs::parse_inputs,
        type_checker::TypeChecker,
    };

    #[test]
    fn test_public_output_constraint() -> miette::Result<()> {
        let code = r"fn main(pub public_input: Field, private_input: Field) -> Field {
            let xx = private_input + public_input;
            assert_eq(xx, 2);
            let yy = xx + 6;
            return yy;
        }";

        let mut sources = Sources::new();
        let mut tast = TypeChecker::new();
        let this_module = None;
        let _node_id = typecheck_next_file(
            &mut tast,
            this_module,
            &mut sources,
            "inline_test_output.no".to_string(),
            code.to_owned(),
            0,
        )
        .unwrap();

        let kimchi_vesta = KimchiVesta::new(false);
        let compiled_circuit = compile(&sources, tast, kimchi_vesta)?;

        let (prover_index, _) = compiled_circuit.compile_to_indexes().unwrap();

        // parse inputs
        let public_inputs = parse_inputs(r#"{"public_input": "1"}"#).unwrap();
        let private_inputs = parse_inputs(r#"{"private_input": "1"}"#).unwrap();

        let public_input_val: u64 = public_inputs
            .0
            .get("public_input")
            .unwrap()
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let output_val = 8;

        let generated_witness = generate_witness(
            &prover_index.compiled_circuit,
            &sources,
            public_inputs,
            private_inputs,
        )?;

        assert_eq!(
            generated_witness.full_public_inputs,
            vec![
                VestaField::from(output_val),
                VestaField::from(public_input_val),
            ]
        );

        let mut full_public_inputs = generated_witness.full_public_inputs.clone();
        let mut witness = generated_witness.all_witness.to_kimchi_witness();
        let gate_count = prover_index.compiled_circuit.circuit.backend.gates.len();
        assert_eq!(gate_count, 6);
        assert_eq!(witness[0][0], VestaField::from(output_val));
        // this is the gate contains the output var
        let output_row = 0;
        assert_eq!(witness[0][output_row], VestaField::from(output_val));
        // this is the assert_eq gate which contains result var and the output var
        let result_row = gate_count - 1;
        assert_eq!(witness[0][result_row], VestaField::from(output_val));

        // should pass the sanity check
        prover_index
            .index
            .verify(&witness, &full_public_inputs)
            .unwrap();

        // first fradulent attempt: modifying one of the public output values
        // attemp to modify the output value
        let invalid_output = VestaField::from(output_val + 1);
        full_public_inputs[1] = invalid_output;

        // this is the gate for the output value
        witness[0][output_row] = invalid_output;

        // verify the witness
        let result = prover_index.index.verify(&witness, &full_public_inputs);

        assert!(result.is_err(), "should failed with incorrect output");

        // should fail the wire check, since the output value at the end (row 5) is different from this change (row 1)
        match result.unwrap_err() {
            GateError::DisconnectedWires(w1, w2) => {
                assert_eq!(w1.row, output_row);
                assert_eq!(w1.col, 0);
                assert_eq!(w2.row, result_row);
                assert_eq!(w2.col, 0);
            }
            _ => panic!("Expected DisconnectedWires error"),
        }

        // second fradulent attempt: sync the value for all the output vars in all the gates
        witness[0][result_row] = invalid_output;

        let result = prover_index.index.verify(&witness, &full_public_inputs);

        assert!(result.is_err(), "should failed with incorrect output");

        // should fail constraint check for the output value
        match result.unwrap_err() {
            GateError::Custom { row, err } => {
                assert_eq!(row, output_row);
                assert_eq!(err, "generic: incorrect gate");
            }
            _ => panic!("Expected incorrect generic gate error"),
        }

        Ok(())
    }
}

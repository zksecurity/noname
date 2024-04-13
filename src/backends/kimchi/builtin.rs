use std::sync::Arc;

use kimchi::{circuits::polynomials::poseidon::{POS_ROWS_PER_HASH, ROUNDS_PER_ROW}, mina_poseidon::{
    constants::{PlonkSpongeConstantsKimchi, SpongeConstants},
    pasta::fp_kimchi::params, permutation::full_round,
}};

use ark_ff::Zero;

use crate::{
    backends::{self, Backend},
    circuit_writer::{CircuitWriter, GateKind, VarInfo},
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::types::TyKind,
    var::{ConstOrCell, Value, Var},
};

use super::{KimchiField, KimchiVesta};

pub fn poseidon(
    circuit: &mut CircuitWriter<KimchiVesta>,
    vars: &[VarInfo<KimchiField>],
    span: Span,
) -> Result<Option<Var<KimchiField>>> {
    //
    // sanity checks
    //

    // only one [Var] is passed
    assert_eq!(vars.len(), 1);
    let var_info = &vars[0];

    // an array of length 2
    match &var_info.typ {
        Some(TyKind::Array(el_typ, 2)) => {
            assert!(matches!(&**el_typ, TyKind::Field | TyKind::BigInt));
        }
        _ => panic!("wrong type for input to poseidon"),
    };

    // extract the values
    let input = &var_info.var;
    assert_eq!(input.len(), 2);

    // hashing a full-constant input is not a good idea
    if input[0].is_const() && input[1].is_const() {
        return Err(Error::new(
            "constraint-generation",
            ErrorKind::UnexpectedError("cannot hash a full-constant input"),
            span,
        ));
    }

    // IMPORTANT: time to constrain any constants
    let mut cells = vec![];
    for const_or_cell in &input.cvars {
        match const_or_cell {
            ConstOrCell::Const(cst) => {
                let cell = circuit.backend.add_constant(
                    Some("encoding constant input to poseidon"),
                    *cst,
                    span,
                );
                cells.push(cell);
            }
            ConstOrCell::Cell(cell) => cells.push(*cell),
        }
    }

    // get constants needed for poseidon
    // TODO: should import directly from kimchi lib
    let poseidon_params = params();

    let rc = &poseidon_params.round_constants;
    let width = PlonkSpongeConstantsKimchi::SPONGE_WIDTH;

    // pad the input (for the capacity)
    let zero_var = circuit.backend.add_constant(
        Some("encoding constant 0 for the capacity of poseidon"),
        KimchiField::zero(),
        span,
    );
    cells.push(zero_var);

    let mut states = vec![cells.clone()];

    // 0..11
    for row in 0..POS_ROWS_PER_HASH {
        let offset = row * ROUNDS_PER_ROW; // row * 5

        // 0..5
        for i in 0..ROUNDS_PER_ROW {
            let mut new_state = vec![];

            let prev_0 = states[states.len() - 1][0];
            let prev_1 = states[states.len() - 1][1];
            let prev_2 = states[states.len() - 1][2];

            for col in 0..3 {
                // create each variable
                let var = circuit.backend.new_internal_var(
                    Value::Hint(Arc::new(move |compiler, env| {
                        let x1 = compiler.compute_var(env, prev_0)?;
                        let x2 = compiler.compute_var(env, prev_1)?;
                        let x3 = compiler.compute_var(env, prev_2)?;

                        let mut acc = vec![x1, x2, x3];

                        // Do one full round on the previous value
                        full_round::<KimchiField, PlonkSpongeConstantsKimchi>(
                            &params(),
                            &mut acc,
                            offset + i,
                        );

                        Ok(acc[col])
                    })),
                    span,
                );

                new_state.push(var);
            }

            states.push(new_state);
        }

        let coeffs = (0..backends::kimchi::NUM_REGISTERS)
            .map(|i| rc[offset + (i / width)][i % width])
            .collect();

        let vars = vec![
            Some(states[offset][0]),
            Some(states[offset][1]),
            Some(states[offset][2]),
            Some(states[offset + 4][0]),
            Some(states[offset + 4][1]),
            Some(states[offset + 4][2]),
            Some(states[offset + 1][0]),
            Some(states[offset + 1][1]),
            Some(states[offset + 1][2]),
            Some(states[offset + 2][0]),
            Some(states[offset + 2][1]),
            Some(states[offset + 2][2]),
            Some(states[offset + 3][0]),
            Some(states[offset + 3][1]),
            Some(states[offset + 3][2]),
        ];

        circuit.backend.add_gate(
            "uses a poseidon gate to constrain 5 rounds of poseidon",
            GateKind::Poseidon,
            vars,
            coeffs,
            span,
        );
    }

    let final_state = &states[states.len() - 1];
    let final_row = vec![
        Some(final_state[0]),
        Some(final_state[1]),
        Some(final_state[2]),
    ];

    // zero gate to store the result
    circuit.backend.add_gate(
        "uses a zero gate to store the output of poseidon",
        GateKind::Zero,
        final_row.clone(),
        vec![],
        span,
    );

    let vars = final_row
        .iter()
        .flatten()
        .cloned()
        .map(ConstOrCell::Cell)
        .collect();

    Ok(Some(Var::new(vars, span)))
}
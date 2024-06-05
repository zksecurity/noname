use std::sync::Arc;

use ark_ff::Zero;
use kimchi::circuits::polynomials::poseidon::{POS_ROWS_PER_HASH, ROUNDS_PER_ROW};
use kimchi::mina_poseidon::constants::{PlonkSpongeConstantsKimchi, SpongeConstants};
use kimchi::mina_poseidon::permutation::full_round;

use super::{KimchiCellVar, KimchiVesta, VestaField};
use crate::backends::kimchi::NUM_REGISTERS;
use crate::backends::Backend;

use crate::{
    circuit_writer::{CircuitWriter, GateKind, VarInfo},
    constants::Span,
    error::{ErrorKind, Result},
    parser::types::TyKind,
    var::{ConstOrCell, Value, Var},
};

pub fn poseidon(
    compiler: &mut CircuitWriter<KimchiVesta>,
    vars: &[VarInfo<VestaField, KimchiCellVar>],
    span: Span,
) -> Result<Option<Var<VestaField, KimchiCellVar>>> {
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
        return Err(compiler.error(
            ErrorKind::UnexpectedError("cannot hash a full-constant input"),
            span,
        ));
    }

    // IMPORTANT: time to constrain any constants
    let mut cells = vec![];
    for const_or_cell in &input.cvars {
        match const_or_cell {
            ConstOrCell::Const(cst) => {
                let cell = compiler.backend.add_constant(
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
    let poseidon_params = kimchi::mina_poseidon::pasta::fp_kimchi::params();

    let rc = &poseidon_params.round_constants;
    let width = PlonkSpongeConstantsKimchi::SPONGE_WIDTH;

    // pad the input (for the capacity)
    let zero_var = compiler.backend.add_constant(
        Some("encoding constant 0 for the capacity of poseidon"),
        VestaField::zero(),
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
                let var = compiler.backend.new_internal_var(
                    Value::Hint(Arc::new(move |backend, env| {
                        let x1 = backend.compute_var(env, &prev_0)?;
                        let x2 = backend.compute_var(env, &prev_1)?;
                        let x3 = backend.compute_var(env, &prev_2)?;

                        let mut acc = vec![x1, x2, x3];

                        // Do one full round on the previous value
                        full_round::<VestaField, PlonkSpongeConstantsKimchi>(
                            &kimchi::mina_poseidon::pasta::fp_kimchi::params(),
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

        let coeffs = (0..NUM_REGISTERS)
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

        compiler.backend.add_gate(
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
    compiler.backend.add_gate(
        "uses a zero gate to store the output of poseidon",
        GateKind::Zero,
        final_row.clone(),
        vec![],
        span,
    );

    let vars = final_row
        .iter()
        .flatten()
        .copied()
        .map(ConstOrCell::Cell)
        .collect();

    Ok(Some(Var::new(vars, span)))
}

use ark_ff::Zero;
use kimchi::{
    circuits::polynomials::poseidon::{POS_ROWS_PER_HASH, ROUNDS_PER_ROW},
    oracle::{
        self,
        constants::{PlonkSpongeConstantsKimchi, SpongeConstants},
        permutation::full_round,
    },
};

use crate::{
    circuit_writer::{CircuitWriter, ConstOrCell, GateKind, Value, Var},
    constants::{self, Field, Span},
    imports::FuncType,
};

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub const CRYPTO_FNS: [(&str, FuncType); 1] = [(POSEIDON_FN, poseidon)];

pub fn poseidon(compiler: &mut CircuitWriter, vars: &[Var], span: Span) -> Option<Var> {
    // double check input
    assert_eq!(vars.len(), 1);
    let input = vars[0].value.clone();

    assert_eq!(input.len(), 2); // size of poseidon input

    // hashing a full-constant input is not a good idea
    if matches!(
        (&input[0], &input[1]),
        (ConstOrCell::Const(_), ConstOrCell::Const(_))
    ) {
        panic!("cannot hash a full-constant input (TODO: better error)");
    }

    // time to constrain the input if they're constants
    let mut cells = vec![];
    for const_or_cell in input {
        match const_or_cell {
            ConstOrCell::Const(cst) => {
                let cell = cst.constrain(Some("encoding constant input to poseidon"), compiler);
                cells.push(cell);
            }
            ConstOrCell::Cell(cell) => cells.push(cell),
        }
    }

    // get constants needed for poseidon
    let poseidon_params = oracle::pasta::fp_kimchi::params();

    let rc = &poseidon_params.round_constants;
    let width = PlonkSpongeConstantsKimchi::SPONGE_WIDTH;

    // pad the input (for the capacity)
    let zero_var = compiler.add_constant(
        Some("encoding constant 0 for the capacity of poseidon"),
        Field::zero(),
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
                let var = compiler.new_internal_var(
                    Value::Hint(Box::new(move |compiler, env| {
                        let x1 = compiler.compute_var(env, prev_0)?;
                        let x2 = compiler.compute_var(env, prev_1)?;
                        let x3 = compiler.compute_var(env, prev_2)?;

                        let mut acc = vec![x1, x2, x3];

                        // Do one full round on the previous value
                        full_round::<Field, PlonkSpongeConstantsKimchi>(
                            &oracle::pasta::fp_kimchi::params(),
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

        let coeffs = (0..constants::NUM_REGISTERS)
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

        compiler.add_gate(
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
    compiler.add_gate(
        "uses a zero gate to store the output of poseidon",
        GateKind::DoubleGeneric,
        final_row.clone(),
        vec![],
        span,
    );

    //    states.borrow_mut().pop().unwrap();
    let vars = final_row.iter().flatten().cloned().collect();
    Some(Var::new_vars(vars, span))
}

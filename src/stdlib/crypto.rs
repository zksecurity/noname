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
    circuit_writer::{CircuitWriter, GateKind, Value, Var},
    constants::{self, Span},
    field::Field,
    imports::FuncType,
};

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub const CRYPTO_FNS: [(&str, FuncType); 1] = [(POSEIDON_FN, poseidon)];

pub fn poseidon(compiler: &mut CircuitWriter, vars: &[Var], span: Span) -> Option<Var> {
    // double check input
    assert_eq!(vars.len(), 1);
    let mut input = match vars[0].circuit_var() {
        None => unimplemented!(),
        Some(cvar) => cvar.vars,
    };
    assert_eq!(input.len(), 2);

    // get constants needed for poseidon
    let poseidon_params = oracle::pasta::fp_kimchi::params();

    let rc = &poseidon_params.round_constants;
    let width = PlonkSpongeConstantsKimchi::SPONGE_WIDTH;

    // pad the input (for the capacity)
    let zero = compiler.add_constant(Field::zero(), span);
    input.push(zero);

    let mut states = vec![input.clone()];

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
    Some(Var::new_circuit_var(vars, span))
}

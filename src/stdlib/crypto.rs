use ark_ff::Zero;
use kimchi::circuits::polynomials::poseidon::{POS_ROWS_PER_HASH, ROUNDS_PER_ROW};
use kimchi::mina_poseidon::constants::{PlonkSpongeConstantsKimchi, SpongeConstants};
use kimchi::mina_poseidon::permutation::full_round;

use crate::backends::kimchi::KimchiVesta;
use crate::backends::Backend;
use crate::imports::FnKind;
use crate::lexer::Token;
use crate::parser::types::FnSig;
use crate::parser::ParserCtx;
use crate::type_checker::FnInfo;
use crate::{
    circuit_writer::{CircuitWriter, GateKind, VarInfo},
    constants::{self, Field, Span},
    error::{ErrorKind, Result},
    parser::types::TyKind,
    var::{ConstOrCell, Value, Var},
};

const POSEIDON_FN: &str = "poseidon(input: [Field; 2]) -> [Field; 3]";

pub const CRYPTO_SIGS: &[&str] = &[POSEIDON_FN];

pub fn get_crypto_fn<B: Backend>(name: &str) -> Option<FnInfo<B>> {
    let ctx = &mut ParserCtx::default();
    let mut tokens = Token::parse(0, name).unwrap();
    let sig = FnSig::parse(ctx, &mut tokens).unwrap();

    let fn_handle = match name {
        POSEIDON_FN => B::poseidon(),
        _ => return None,
    };

    Some(FnInfo {
        kind: FnKind::BuiltIn(sig, fn_handle),
        span: Span::default(),
    })
}

/// a function returns crypto functions
pub fn crypto_fns<B: Backend>() -> Vec<FnInfo<B>> {
    CRYPTO_SIGS
        .iter()
        .map(|sig| get_crypto_fn(sig).unwrap())
        .collect()
}

pub fn poseidon(compiler: &mut CircuitWriter<KimchiVesta>, vars: &[VarInfo<Field>], span: Span) -> Result<Option<Var<Field>>> {
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
                let cell =
                    compiler.add_constant(Some("encoding constant input to poseidon"), *cst, span);
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

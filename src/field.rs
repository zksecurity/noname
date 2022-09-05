use crate::{
    boolean,
    circuit_writer::{CircuitWriter, ConstOrCell, Constant, GateKind, Value, Var},
    constants::{Field, Span},
};

use ark_ff::{One, Zero};

use std::ops::Neg;

/// Adds two field elements
pub fn add(compiler: &mut CircuitWriter, lhs: Var, rhs: Var, span: Span) -> Var {
    let zero = Field::zero();
    let one = Field::one();

    // sanity check
    assert_eq!(rhs.len(), 1);
    assert_eq!(lhs.len(), 1);

    match (&lhs[0], &rhs[0]) {
        // 2 constants
        (
            ConstOrCell::Const(Constant { value: lhs, .. }),
            ConstOrCell::Const(Constant { value: rhs, .. }),
        ) => Var::new_constant(
            Constant {
                value: *lhs + *rhs,
                span,
            },
            span,
        ),

        // const and a var
        (ConstOrCell::Const(Constant { value: cst, .. }), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(Constant { value: cst, .. })) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                // TODO: that span is incorrect, it should come from lhs or rhs...
                return Var::new_vars(vec![*cvar], span);
            }

            // create a new variable to store the result
            let res =
                compiler.new_internal_var(Value::LinearCombination(vec![(one, *cvar)], *cst), span);

            // create a gate to store the result
            // TODO: we should use an add_generic function that takes advantage of the double generic gate
            compiler.add_gate(
                "add a constant with a variable",
                GateKind::DoubleGeneric,
                vec![Some(*cvar), None, Some(res)],
                vec![one, zero, one.neg(), zero, *cst],
                span,
            );

            Var::new_vars(vec![res], span)
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(
                    vec![(Field::one(), *lhs), (Field::one(), *rhs)],
                    Field::zero(),
                ),
                span,
            );

            // create a gate to store the result
            compiler.add_gate(
                "add two variables together",
                GateKind::DoubleGeneric,
                vec![Some(*lhs), Some(*rhs), Some(res)],
                vec![Field::one(), Field::one(), Field::one().neg()],
                span,
            );

            Var::new_vars(vec![res], span)
        }
    }
}

/// Subtracts two variables, we only support variables that are of length 1.
pub fn sub_vars(compiler: &mut CircuitWriter, lhs: Var, rhs: Var, span: Span) -> Var {
    // sanity check
    assert_eq!(rhs.len(), 1);
    assert_eq!(lhs.len(), 1);

    sub_cells(compiler, &lhs[0], &rhs[0], span)
}

/// Subtracts two cells lhs - rhs
pub fn sub_cells(
    compiler: &mut CircuitWriter,
    lhs: &ConstOrCell,
    rhs: &ConstOrCell,
    span: Span,
) -> Var {
    let zero = Field::zero();
    let one = Field::one();

    match (lhs, rhs) {
        // const1 - const2
        (
            ConstOrCell::Const(Constant { value: lhs, .. }),
            ConstOrCell::Const(Constant { value: rhs, .. }),
        ) => Var::new_constant(
            Constant {
                value: *lhs - *rhs,
                span,
            },
            span,
        ),

        // const - var
        (ConstOrCell::Const(Constant { value: cst, .. }), ConstOrCell::Cell(cvar)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one.neg(), *cvar)], *cst),
                span,
            );

            // create a gate to store the result
            compiler.add_gate(
                "constant - variable",
                GateKind::DoubleGeneric,
                vec![Some(*cvar), None, Some(res)],
                // cst - cvar - out = 0
                vec![one.neg(), zero, one.neg(), zero, *cst],
                span,
            );

            Var::new_vars(vec![res], span)
        }

        // var - const
        (ConstOrCell::Cell(cvar), ConstOrCell::Const(Constant { value: cst, .. })) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                // TODO: that span is incorrect, it should come from lhs or rhs...
                return Var::new_vars(vec![*cvar], span);
            }

            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, *cvar)], cst.neg()),
                span,
            );

            // create a gate to store the result
            // TODO: we should use an add_generic function that takes advantage of the double generic gate
            compiler.add_gate(
                "variable - constant",
                GateKind::DoubleGeneric,
                vec![Some(*cvar), None, Some(res)],
                // var - cst - out = 0
                vec![one, zero, one.neg(), zero, cst.neg()],
                span,
            );

            Var::new_vars(vec![res], span)
        }

        // lhs - rhs
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, *lhs), (one.neg(), *rhs)], zero),
                span,
            );

            // create a gate to store the result
            compiler.add_gate(
                "var1 - var2",
                GateKind::DoubleGeneric,
                vec![Some(*lhs), Some(*rhs), Some(res)],
                // lhs - rhs - out = 0
                vec![one, one.neg(), one.neg()],
                span,
            );

            Var::new_vars(vec![res], span)
        }
    }
}

/// This takes variables that can be anything, and returns a boolean
// TODO: so perhaps it's not really relevant in this file?
pub fn equal_vars(compiler: &mut CircuitWriter, lhs: Var, rhs: Var, span: Span) -> Var {
    // sanity check
    assert_eq!(lhs.len(), rhs.len());

    // create an accumulator
    let one = Field::one();

    let acc = compiler.add_constant(
        Some("start accumulator at 1 for the equality check"),
        one,
        span,
    );
    let mut acc = Var::new_vars(vec![acc], span);

    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
        let res = equal_cells(compiler, lhs, rhs, span);

        // AND the result
        acc = boolean::and(compiler, res, acc, span);
    }

    acc
}

/// Returns a new variable set to 1 if x1 is equal to x2, 0 otherwise.
fn equal_cells(compiler: &mut CircuitWriter, x1: ConstOrCell, x2: ConstOrCell, span: Span) -> Var {
    // These four constraints are enough:
    //
    // 1. `diff = x2 - x1`
    // 2. `one_minus_res + res = 1`
    // 3. `res * diff = 0`
    // 4. `diff_inv * diff = one_minus_res`
    //
    // To prove this, it suffices to prove that:
    //
    // a. `diff = 0 => res = 1`.
    // b. `diff != 0 => res = 0`.
    //
    // Proof:
    //
    // a. if `diff = 0`,
    //      then using (4) `one_minus_res = 0`,
    //      then using (2) `res = 1`
    //
    // b. if `diff != 0`
    //      then using (3) `res = 0`
    //

    let zero = Field::zero();
    let one = Field::one();

    match (x1, x2) {
        // two constants
        (ConstOrCell::Const(x1), ConstOrCell::Const(x2)) => {
            let res = if x1.value == x2.value {
                one
            } else {
                Field::zero()
            };
            Var::new_constant(Constant { value: res, span }, span)
        }

        (x1, x2) => {
            let x1 = match x1 {
                ConstOrCell::Const(cst) => compiler.add_constant(
                    Some("encode the lhs constant of the equality check in the circuit"),
                    cst.value,
                    span,
                ),
                ConstOrCell::Cell(cvar) => cvar,
            };

            let x2 = match x2 {
                ConstOrCell::Const(cst) => compiler.add_constant(
                    Some("encode the rhs constant of the equality check in the circuit"),
                    cst.value,
                    span,
                ),
                ConstOrCell::Cell(cvar) => cvar,
            };

            // compute the result
            let res = compiler.new_internal_var(
                Value::Hint(Box::new(move |compiler, env| {
                    let x1 = compiler.compute_var(env, x1)?;
                    let x2 = compiler.compute_var(env, x2)?;
                    if x1 == x2 {
                        Ok(Field::one())
                    } else {
                        Ok(Field::zero())
                    }
                })),
                span,
            );

            // 1. diff = x2 - x1
            let diff = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, x2), (one.neg(), x1)], zero),
                span,
            );

            compiler.add_gate(
                "constraint #1 for the equals gadget (x2 - x1 - diff = 0)",
                GateKind::DoubleGeneric,
                vec![Some(x2), Some(x1), Some(diff)],
                // x2 - x1 - diff = 0
                vec![one, one.neg(), one.neg()],
                span,
            );

            // 2. one_minus_res = 1 - res
            let one_minus_res = compiler
                .new_internal_var(Value::LinearCombination(vec![(one.neg(), res)], one), span);

            compiler.add_gate(
                "constraint #2 for the equals gadget (one_minus_res + res - 1 = 0)",
                GateKind::DoubleGeneric,
                vec![Some(one_minus_res), Some(res)],
                // we constrain one_minus_res + res - 1 = 0
                // so that we can encode res and wire it elsewhere
                // (and not -res)
                vec![one, one, zero, zero, one.neg()],
                span,
            );

            // 3. res * diff = 0
            compiler.add_gate(
                "constraint #3 for the equals gadget (res * diff = 0)",
                GateKind::DoubleGeneric,
                vec![Some(res), Some(diff)],
                // res * diff = 0
                vec![zero, zero, zero, one],
                span,
            );

            // 4. diff_inv * diff = one_minus_res
            let diff_inv = compiler.new_internal_var(Value::Inverse(diff), span);

            compiler.add_gate(
                "constraint #4 for the equals gadget (diff_inv * diff = one_minus_res)",
                GateKind::DoubleGeneric,
                vec![Some(diff_inv), Some(diff), Some(one_minus_res)],
                // diff_inv * diff - one_minus_res = 0
                vec![zero, zero, one.neg(), one],
                span,
            );

            Var::new_vars(vec![res], span)
        }
    }
}

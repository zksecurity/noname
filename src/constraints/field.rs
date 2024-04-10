use crate::{
    backends::Backend,
    circuit_writer::CircuitWriter,
    constants::Span,
    var::{ConstOrCell, Value, Var},
};

use super::boolean;

use ark_ff::{One, Zero};

use std::{ops::Neg, sync::Arc};

/// Adds two field elements
pub fn add<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field>,
    rhs: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    let zero = B::Field::zero();
    let one = B::Field::one();

    match (lhs, rhs) {
        // 2 constants
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs + *rhs, span),

        // const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                // TODO: that span is incorrect, it should come from lhs or rhs...
                return Var::new_var(*cvar, span);
            }

            // create a new variable to store the result
            let res =
                compiler.new_internal_var(Value::LinearCombination(vec![(one, *cvar)], *cst), span);

            // create a gate to store the result
            compiler.backend.add_constraint(
                "add a constant with a variable",
                vec![Some(*cvar), None, Some(res)],
                vec![one, zero, one.neg(), zero, *cst],
                span,
            );

            Var::new_var(res, span)
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(
                    vec![(B::Field::one(), *lhs), (B::Field::one(), *rhs)],
                    B::Field::zero(),
                ),
                span,
            );

            compiler.backend.add_constraint(
                "add two variables together",
                vec![Some(*lhs), Some(*rhs), Some(res)],
                vec![B::Field::one(), B::Field::one(), B::Field::one().neg()],
                span,
            );

            Var::new_var(res, span)
        }
    }
}

/// Subtracts two variables, we only support variables that are of length 1.
pub fn sub<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field>,
    rhs: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    let zero = B::Field::zero();
    let one = B::Field::one();

    match (lhs, rhs) {
        // const1 - const2
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs - *rhs, span),

        // const - var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one.neg(), *cvar)], *cst),
                span,
            );

            // create a gate to store the result
            compiler.backend.add_constraint(
                "constant - variable",
                vec![Some(*cvar), None, Some(res)],
                // cst - cvar - out = 0
                vec![one.neg(), zero, one.neg(), zero, *cst],
                span,
            );

            Var::new_var(res, span)
        }

        // var - const
        (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                // TODO: that span is incorrect, it should come from lhs or rhs...
                return Var::new_var(*cvar, span);
            }

            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, *cvar)], cst.neg()),
                span,
            );

            // create a gate to store the result
            // TODO: we should use an add_generic function that takes advantage of the double generic gate
            compiler.backend.add_constraint(
                "variable - constant",
                vec![Some(*cvar), None, Some(res)],
                // var - cst - out = 0
                vec![one, zero, one.neg(), zero, cst.neg()],
                span,
            );

            Var::new_var(res, span)
        }

        // lhs - rhs
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, *lhs), (one.neg(), *rhs)], zero),
                span,
            );

            // create a gate to store the result
            compiler.backend.add_constraint(
                "var1 - var2",
                vec![Some(*lhs), Some(*rhs), Some(res)],
                // lhs - rhs - out = 0
                vec![one, one.neg(), one.neg()],
                span,
            );

            Var::new_var(res, span)
        }
    }
}

/// Multiplies two field elements
pub fn mul<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field>,
    rhs: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    let zero = B::Field::zero();
    let one = B::Field::one();

    match (lhs, rhs) {
        // 2 constants
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs * *rhs, span),

        // const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                let zero = compiler.add_constant(
                    Some("encoding zero for the result of 0 * var"),
                    B::Field::zero(),
                    span,
                );
                return Var::new_var(zero, span);
            }

            // create a new variable to store the result
            let res = compiler.new_internal_var(Value::Scale(*cst, *cvar), span);

            // create a gate to store the result
            // TODO: we should use an add_generic function that takes advantage of the double generic gate
            compiler.backend.add_constraint(
                "add a constant with a variable",
                vec![Some(*cvar), None, Some(res)],
                vec![*cst, zero, one.neg(), zero, *cst],
                span,
            );

            Var::new_var(res, span)
        }

        // everything is a var
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // create a new variable to store the result
            let res = compiler.new_internal_var(Value::Mul(*lhs, *rhs), span);

            // create a gate to store the result
            compiler.backend.add_constraint(
                "add two variables together",
                vec![Some(*lhs), Some(*rhs), Some(res)],
                vec![zero, zero, one.neg(), one],
                span,
            );

            Var::new_var(res, span)
        }
    }
}

/// This takes variables that can be anything, and returns a boolean
// TODO: so perhaps it's not really relevant in this file?
pub fn equal<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &Var<B::Field>,
    rhs: &Var<B::Field>,
    span: Span,
) -> Var<B::Field> {
    // sanity check
    assert_eq!(lhs.len(), rhs.len());

    if lhs.len() == 1 {
        return equal_cells(compiler, &lhs[0], &rhs[0], span);
    }

    // create an accumulator
    let one = B::Field::one();

    let acc = compiler.add_constant(
        Some("start accumulator at 1 for the equality check"),
        one,
        span,
    );
    let mut acc = Var::new_var(acc, span);

    for (l, r) in lhs.cvars.iter().zip(&rhs.cvars) {
        let res = equal_cells(compiler, l, r, span);
        acc = boolean::and(compiler, &res[0], &acc[0], span);
    }

    acc
}

/// Returns a new variable set to 1 if x1 is equal to x2, 0 otherwise.
fn equal_cells<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    x1: &ConstOrCell<B::Field>,
    x2: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
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

    let zero = B::Field::zero();
    let one = B::Field::one();

    match (x1, x2) {
        // two constants
        (ConstOrCell::Const(x1), ConstOrCell::Const(x2)) => {
            let res = if x1 == x2 { one } else { B::Field::zero() };
            Var::new_constant(res, span)
        }

        (x1, x2) => {
            let x1 = match x1 {
                ConstOrCell::Const(cst) => compiler.add_constant(
                    Some("encode the lhs constant of the equality check in the circuit"),
                    *cst,
                    span,
                ),
                ConstOrCell::Cell(cvar) => *cvar,
            };

            let x2 = match x2 {
                ConstOrCell::Const(cst) => compiler.add_constant(
                    Some("encode the rhs constant of the equality check in the circuit"),
                    *cst,
                    span,
                ),
                ConstOrCell::Cell(cvar) => *cvar,
            };

            // compute the result
            let res = compiler.new_internal_var(
                Value::Hint(Arc::new(move |compiler, env| {
                    let x1 = compiler.compute_var(env, x1)?;
                    let x2 = compiler.compute_var(env, x2)?;
                    if x1 == x2 {
                        Ok(B::Field::one())
                    } else {
                        Ok(B::Field::zero())
                    }
                })),
                span,
            );

            // 1. diff = x2 - x1
            let diff = compiler.new_internal_var(
                Value::LinearCombination(vec![(one, x2), (one.neg(), x1)], zero),
                span,
            );

            compiler.backend.add_constraint(
                "constraint #1 for the equals gadget (x2 - x1 - diff = 0)",
                vec![Some(x2), Some(x1), Some(diff)],
                // x2 - x1 - diff = 0
                vec![one, one.neg(), one.neg()],
                span,
            );

            // 2. one_minus_res = 1 - res
            let one_minus_res = compiler
                .new_internal_var(Value::LinearCombination(vec![(one.neg(), res)], one), span);

            compiler.backend.add_constraint(
                "constraint #2 for the equals gadget (one_minus_res + res - 1 = 0)",
                vec![Some(one_minus_res), Some(res)],
                // we constrain one_minus_res + res - 1 = 0
                // so that we can encode res and wire it elsewhere
                // (and not -res)
                vec![one, one, zero, zero, one.neg()],
                span,
            );

            // 3. res * diff = 0
            compiler.backend.add_constraint(
                "constraint #3 for the equals gadget (res * diff = 0)",
                vec![Some(res), Some(diff)],
                // res * diff = 0
                vec![zero, zero, zero, one],
                span,
            );

            // 4. diff_inv * diff = one_minus_res
            let diff_inv = compiler.new_internal_var(Value::Inverse(diff), span);

            compiler.backend.add_constraint(
                "constraint #4 for the equals gadget (diff_inv * diff = one_minus_res)",
                vec![Some(diff_inv), Some(diff), Some(one_minus_res)],
                // diff_inv * diff - one_minus_res = 0
                vec![zero, zero, one.neg(), one],
                span,
            );

            Var::new_var(res, span)
        }
    }
}

pub fn if_else<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cond: &Var<B::Field>,
    then_: &Var<B::Field>,
    else_: &Var<B::Field>,
    span: Span,
) -> Var<B::Field> {
    assert_eq!(cond.len(), 1);
    assert_eq!(then_.len(), else_.len());

    let cond = cond[0];

    let mut vars = vec![];

    for (then_, else_) in then_.cvars.iter().zip(&else_.cvars) {
        let var = if_else_inner(compiler, &cond, then_, else_, span);
        vars.push(var[0]);
    }

    Var::new(vars, span)
}

pub fn if_else_inner<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cond: &ConstOrCell<B::Field>,
    then_: &ConstOrCell<B::Field>,
    else_: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    // we need to constrain:
    //
    // * res = (1 - cond) * else + cond * then
    //

    // if cond is constant, easy
    let cond_cell = match cond {
        ConstOrCell::Const(cond) => {
            if cond.is_one() {
                return Var::new_cvar(*then_, span);
            } else {
                return Var::new_cvar(*else_, span);
            }
        }
        ConstOrCell::Cell(cond) => *cond,
    };

    match (&then_, &else_) {
        // if the branches are constant,
        // we can create the following constraints:
        //
        // res = (1 - cond) * else + cond * then
        //
        // translates to
        //
        // cond_then = cond * then
        // temp = (1 - cond) * else =>
        //      - either
        //          - one_minus_cond = 1 - cond
        //          - one_minus_cond * else
        //      - or
        //          - cond_else = cond * else
        //          - else - cond_else
        // res - temp + cond_then = 0
        // res - X = 0
        //
        (ConstOrCell::Const(_), ConstOrCell::Const(_)) => {
            let cond_then = mul(compiler, then_, cond, span);
            let one = ConstOrCell::Const(B::Field::one());
            let one_minus_cond = sub(compiler, &one, cond, span);
            let temp = mul(compiler, &one_minus_cond[0], else_, span);
            add(compiler, &cond_then[0], &temp[0], span)
        }

        // if one of them is a var
        //
        // res = (1 - cond) * else + cond * then
        //
        // translates to
        //
        // cond_then = cond * then
        // temp = (1 - cond) * else =>
        //      - either
        //          - one_minus_cond = 1 - cond
        //          - one_minus_cond * else
        //      - or
        //          - cond_else = cond * else
        //          - else - cond_else
        // res - temp + cond_then = 0
        // res - X = 0
        //
        _ => {
            //            let cond_inner = cond.clone();
            let then_clone = *then_;
            let else_clone = *else_;

            let res = compiler.new_internal_var(
                Value::Hint(Arc::new(move |compiler, env| {
                    let cond = compiler.compute_var(env, cond_cell)?;
                    let res_var = if cond.is_one() {
                        &then_clone
                    } else {
                        &else_clone
                    };
                    match res_var {
                        ConstOrCell::Const(cst) => Ok(*cst),
                        ConstOrCell::Cell(var) => compiler.compute_var(env, *var),
                    }
                })),
                span,
            );

            let then_m_else = sub(compiler, then_, else_, span)[0]
                .cvar()
                .cloned()
                .unwrap();
            let res_m_else = sub(compiler, &ConstOrCell::Cell(res), else_, span)[0]
                .cvar()
                .cloned()
                .unwrap();

            let zero = B::Field::zero();
            let one = B::Field::one();

            compiler.backend.add_constraint(
                "constraint for ternary operator: cond * (then - else) = res - else",
                vec![Some(cond_cell), Some(then_m_else), Some(res_m_else)],
                // cond * (then - else) = res - else
                vec![zero, zero, one.neg(), one],
                span,
            );

            Var::new_var(res, span)
        }
    }
}

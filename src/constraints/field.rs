use crate::{
    backends::Backend,
    circuit_writer::CircuitWriter,
    constants::Span,
    var::{ConstOrCell, Value, Var},
};

use super::boolean;

use ark_ff::{One, Zero};

use std::ops::Neg;

/// Negates a field element
pub fn neg<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cvar: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
    match cvar {
        crate::var::ConstOrCell::Const(ff) => Var::new_constant(ff.neg(), span),
        crate::var::ConstOrCell::Cell(var) => {
            let res = compiler.backend.neg(var, span);
            Var::new_var(res, span)
        }
    }
}

/// Adds two field elements
pub fn add<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::CellVar>,
    rhs: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
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

            let res = compiler.backend.add_const(cvar, cst, span);

            Var::new_var(res, span)
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            let res = compiler.backend.add(lhs, rhs, span);
            Var::new_var(res, span)
        }
    }
}

/// Subtracts two variables, we only support variables that are of length 1.
pub fn sub<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::CellVar>,
    rhs: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
    let neg_rhs = neg(compiler, rhs, span);
    add(compiler, lhs, &neg_rhs.cvars[0], span)
}

/// Multiplies two field elements
pub fn mul<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::CellVar>,
    rhs: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
    match (lhs, rhs) {
        // 2 constants
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs * *rhs, span),

        // const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                return Var::new_constant(*cst, span);
            }

            let res = compiler.backend.mul_const(cvar, cst, span);
            Var::new_var(res, span)
        }

        // everything is a var
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            let res = compiler.backend.mul(lhs, rhs, span);
            Var::new_var(res, span)
        }
    }
}

/// This takes variables that can be anything, and returns a boolean
// TODO: so perhaps it's not really relevant in this file?
pub fn equal<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &Var<B::Field, B::CellVar>,
    rhs: &Var<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
    // sanity check
    assert_eq!(lhs.len(), rhs.len());

    if lhs.len() == 1 {
        return equal_cells(compiler, &lhs[0], &rhs[0], span);
    }

    // create an accumulator
    let one = B::Field::one();

    let acc = compiler.backend.add_constant(
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
    x1: &ConstOrCell<B::Field, B::CellVar>,
    x2: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
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
                ConstOrCell::Const(cst) => compiler.backend.add_constant(
                    Some("encode the lhs constant of the equality check in the circuit"),
                    *cst,
                    span,
                ),
                ConstOrCell::Cell(cvar) => *cvar,
            };

            let x2 = match x2 {
                ConstOrCell::Const(cst) => compiler.backend.add_constant(
                    Some("encode the rhs constant of the equality check in the circuit"),
                    *cst,
                    span,
                ),
                ConstOrCell::Cell(cvar) => *cvar,
            };

            // compute the result
            let res = compiler.backend.new_internal_var(
                Value::Hint(Arc::new(move |backend, env| {
                    let x1 = backend.compute_var(env, x1)?;
                    let x2 = backend.compute_var(env, x2)?;
                    if x1 == x2 {
                        Ok(B::Field::one())
                    } else {
                        Ok(B::Field::zero())
                    }
                })),
                span,
            );

            // 1. diff = x2 - x1
            let neg_x1 = compiler.backend.neg(&x1, span);
            let diff = compiler.backend.add(&x2, &neg_x1, span);

            // 2. one_minus_res = 1 - res
            let neg_res = compiler.backend.neg(&res, span);
            let one_minus_res = compiler.backend.add_const(&neg_res, &one, span);

            // 3. res * diff = 0
            let res_mul_diff = compiler.backend.mul(&res, &diff, span);

            compiler.backend.assert_eq_const(&res_mul_diff, zero, span);

            // 4. diff_inv * diff = one_minus_res
            let diff_inv = compiler
                .backend
                .new_internal_var(Value::Inverse(diff), span);

            let diff_inv_mul_diff = compiler.backend.mul(&diff_inv, &diff, span);

            compiler
                .backend
                .assert_eq_var(&diff_inv_mul_diff, &one_minus_res, span);

            Var::new_var(res, span)
        }
    }
}

pub fn if_else<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cond: &Var<B::Field, B::CellVar>,
    then_: &Var<B::Field, B::CellVar>,
    else_: &Var<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
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
    cond: &ConstOrCell<B::Field, B::CellVar>,
    then_: &ConstOrCell<B::Field, B::CellVar>,
    else_: &ConstOrCell<B::Field, B::CellVar>,
    span: Span,
) -> Var<B::Field, B::CellVar> {
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

            let res = compiler.backend.new_internal_var(
                Value::Hint(Arc::new(move |backend, env| {
                    let cond = backend.compute_var(env, cond_cell)?;
                    let res_var = if cond.is_one() {
                        &then_clone
                    } else {
                        &else_clone
                    };
                    match res_var {
                        ConstOrCell::Const(cst) => Ok(*cst),
                        ConstOrCell::Cell(var) => backend.compute_var(env, *var),
                    }
                })),
                span,
            );

            let then_m_else = sub(compiler, then_, else_, span)[0];

            // constraint for ternary operator: cond * (then - else) = res - else
            let cond_mul_then_m_else = mul(compiler, cond, &then_m_else, span)[0];
            let res_to_check = add(compiler, &cond_mul_then_m_else, else_, span)[0];
            compiler
                .backend
                .assert_eq_var(&res, res_to_check.cvar().unwrap(), span);

            Var::new_var(res, span)
        }
    }
}

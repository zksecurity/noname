//! Custom types

use std::ops::Neg;

use ark_ff::{Field, One, Zero};

use crate::{
    backends::Backend,
    circuit_writer::CircuitWriter,
    constants::Span,
    var::{ConstOrCell, Value, Var},
};

use super::field;

pub fn is_valid<F: Field>(f: F) -> bool {
    f.is_one() || f.is_zero()
}

pub fn check<B: Backend>(compiler: &mut CircuitWriter<B>, xx: &ConstOrCell<B::Field>, span: Span) {
    let one = B::Field::one();

    match xx {
        ConstOrCell::Const(ff) => assert!(is_valid(*ff)),
        ConstOrCell::Cell(var) => {
            // x * (x - 1)
            let x_1 = compiler.backend.constraint_add_const(var, &one.neg(), span);
            let res = compiler.backend.constraint_mul(var, &x_1, span);
            compiler.backend.constraint_eq_const(&res, B::Field::zero(), span);
        },
    };
}

pub fn and<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field>,
    rhs: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    match (lhs, rhs) {
        // two constants
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs * *rhs, span),

        // constant and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            if cst.is_one() {
                Var::new_var(*cvar, span)
            } else {
                Var::new_constant(*cst, span)
            }
        }

        // two vars
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // lhs * rhs
            let res = compiler.backend.constraint_mul(lhs, rhs, span);

            // check if it is either 1 or 0
            check(compiler, &ConstOrCell::Cell(res), span);

            Var::new_var(res, span)
        }
    }
}

pub fn not<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    var: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    match var {
        ConstOrCell::Const(cst) => {
            let value = if cst.is_one() {
                B::Field::zero()
            } else {
                B::Field::one()
            };

            Var::new_constant(value, span)
        }

        // constant and a var
        ConstOrCell::Cell(_) => {
            let one = B::Field::one();

            // 1 - x
            field::sub(compiler, &ConstOrCell::Const(one), var, span)
        }
    }
}

pub fn or<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field>,
    rhs: &ConstOrCell<B::Field>,
    span: Span,
) -> Var<B::Field> {
    let not_lhs = not(compiler, lhs, span);
    let not_rhs = not(compiler, rhs, span);
    let both_false = and(compiler, &not_lhs[0], &not_rhs[0], span);
    not(compiler, &both_false[0], span)
}

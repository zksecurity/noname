//! Custom types

use std::ops::Neg;

use ark_ff::{One, Zero};

use crate::{
    backends::Backend,
    circuit_writer::CircuitWriter,
    constants::Span,
    var::{ConstOrCell, Value, Var},
};

//todo encapsulate in a struct

pub fn is_valid<B: Backend>(f: B::Field) -> bool {
    f.is_one() || f.is_zero()
}

pub fn check<B: Backend>(compiler: &mut CircuitWriter<B>, xx: &ConstOrCell<B::Field>, span: Span) {
    let zero = B::Field::zero();
    let one = B::Field::one();

    match xx {
        ConstOrCell::Const(ff) => assert!(is_valid::<B>(*ff)),
        ConstOrCell::Cell(var) => compiler.backend.add_generic_gate(
            "constraint to validate a boolean (`x(x-1) = 0`)",
            // x^2 - x = 0
            vec![Some(*var), Some(*var), None],
            vec![one.neg(), zero, zero, one],
            span,
        ),
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
            // create a new variable to store the result
            let res = compiler.backend.new_internal_var(Value::Mul(*lhs, *rhs), span);

            // create a gate to constrain the result
            let zero = B::Field::zero();
            let one = B::Field::one();
            compiler.backend.add_generic_gate(
                "constrain the AND as lhs * rhs",
                vec![Some(*lhs), Some(*rhs), Some(res)],
                vec![zero, zero, one.neg(), one], // mul
                span,
            );

            // return the result
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
        ConstOrCell::Cell(cvar) => {
            let zero = B::Field::zero();
            let one = B::Field::one();

            // create a new variable to store the result
            let lc = Value::LinearCombination(vec![(one.neg(), *cvar)], one); // 1 - X
            let res = compiler.backend.new_internal_var(lc, span);

            // create a gate to constrain the result
            compiler.backend.add_generic_gate(
                "constrain the NOT as 1 - X",
                vec![None, Some(*cvar), Some(res)],
                // we use the constant to do 1 - X
                vec![zero, one.neg(), one.neg(), zero, one],
                span,
            );

            // return the result
            Var::new_var(res, span)
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

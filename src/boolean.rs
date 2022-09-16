//! Custom types

use std::ops::Neg;

use ark_ff::{One, Zero};

use crate::{
    circuit_writer::{CircuitWriter, GateKind},
    constants::{Field, Span},
    var::{ConstOrCell, Constant, Value, Var},
};

pub fn is_valid(f: Field) -> bool {
    f.is_one() || f.is_zero()
}

pub fn and(compiler: &mut CircuitWriter, lhs: Var, rhs: Var, span: Span) -> Var {
    let lhs = lhs
        .const_or_cell()
        .expect("bool.and: lhs is not a constant or cell");
    let rhs = rhs
        .const_or_cell()
        .expect("bool.and: rhs is not a constant or cell");

    match (lhs, rhs) {
        // two constants
        (
            ConstOrCell::Const(Constant { value: lhs, .. }),
            ConstOrCell::Const(Constant { value: rhs, .. }),
        ) => Var::new_constant(Constant::new(*lhs * *rhs, span), span),

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
            let res = compiler.new_internal_var(Value::Mul(*lhs, *rhs), span);

            // create a gate to constrain the result
            let zero = Field::zero();
            let one = Field::one();
            compiler.add_gate(
                "constrain the AND as lhs * rhs",
                GateKind::DoubleGeneric,
                vec![Some(*lhs), Some(*rhs), Some(res)],
                vec![zero, zero, one.neg(), one], // mul
                span,
            );

            // return the result
            Var::new_var(res, span)
        }
    }
}

pub fn neg(compiler: &mut CircuitWriter, var: Var, span: Span) -> Var {
    let var = var
        .const_or_cell()
        .expect("bool.neg: var is not a constant or cell");

    match var {
        ConstOrCell::Const(cst) => {
            let value = if cst.is_one() {
                Field::zero()
            } else {
                Field::one()
            };

            Var::new_constant(Constant { value, ..*cst }, span)
        }

        // constant and a var
        ConstOrCell::Cell(cvar) => {
            let zero = Field::zero();
            let one = Field::one();

            // create a new variable to store the result
            let lc = Value::LinearCombination(vec![(one.neg(), *cvar)], one); // 1 - X
            let res = compiler.new_internal_var(lc, span);

            // create a gate to constrain the result
            compiler.add_gate(
                "constrain the NOT as 1 - X",
                GateKind::DoubleGeneric,
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

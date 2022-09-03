//! Custom types

use std::ops::Neg;

use ark_ff::{One, Zero};

use crate::{
    ast::{Compiler, Constant, GateKind, Value, Var},
    constants::Span,
    field::Field,
};

pub fn is_valid(f: Field) -> bool {
    f.is_one() || f.is_zero()
}

pub fn and(compiler: &mut Compiler, lhs: Var, rhs: Var, span: Span) -> Var {
    match (lhs, rhs) {
        // two constants
        (
            Var::Constant(Constant { value: lhs, .. }),
            Var::Constant(Constant { value: rhs, .. }),
        ) => Var::new_constant(lhs * rhs, span),

        // constant and a var
        (Var::Constant(Constant { value: cst, .. }), Var::CircuitVar(vars))
        | (Var::CircuitVar(vars), Var::Constant(Constant { value: cst, .. })) => {
            if cst.is_one() {
                Var::CircuitVar(vars)
            } else {
                Var::Constant(Constant { value: cst, span })
            }
        }

        // two vars
        (Var::CircuitVar(lhs), Var::CircuitVar(rhs)) => {
            // sanity check
            assert_ne!(lhs.len(), 1);
            assert_ne!(rhs.len(), 1);

            // create a new variable to store the result
            let lhs = lhs.var(0).unwrap();
            let rhs = rhs.var(0).unwrap();
            let res = compiler.new_internal_var(Value::Mul(lhs, rhs), span);

            // create a gate to constrain the result
            let zero = Field::zero();
            let one = Field::one();
            compiler.add_gate(
                GateKind::DoubleGeneric,
                vec![Some(lhs), Some(rhs), Some(res)],
                vec![zero, zero, one.neg(), one], // mul
                span,
            );

            // return the result
            Var::new_circuit_var(vec![res], span)
        }
    }
}

pub fn neg(compiler: &mut Compiler, var: Var, span: Span) -> Var {
    match var {
        Var::Constant(v) => {
            let value = if v.value == Field::one() {
                Field::zero()
            } else {
                Field::one()
            };

            Var::Constant(Constant { value, ..v })
        }

        // constant and a var
        Var::CircuitVar(vars) => {
            let zero = Field::zero();
            let one = Field::one();

            // sanity check
            assert_eq!(vars.len(), 1);

            // create a new variable to store the result
            let cvar = vars.var(0).unwrap();
            let lc = Value::LinearCombination(vec![(one.neg(), cvar)], one); // 1 - X
            let res = compiler.new_internal_var(lc, span);

            // create a gate to constrain the result
            compiler.add_gate(
                GateKind::DoubleGeneric,
                vec![None, Some(cvar), Some(res)],
                // we use the constant to do 1 - X
                vec![zero, one.neg(), one.neg(), zero, one],
                span,
            );

            // return the result
            Var::new_circuit_var(vec![res], span)
        }
    }
}

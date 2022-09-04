use crate::{
    circuit_writer::{CircuitWriter, ConstOrCell, Constant, GateKind, Value, Var},
    constants::{Field, Span},
};

use ark_ff::{One, Zero};

use std::ops::Neg;

/// Adds two field elements
pub fn add(compiler: &mut CircuitWriter, lhs: Var, rhs: Var, span: Span) -> Var {
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
            let res = compiler.new_internal_var(
                Value::LinearCombination(vec![(Field::one(), *cvar)], *cst),
                span,
            );

            // create a gate to store the result
            // TODO: we should use an add_generic function that takes advantage of the double generic gate
            let zero = Field::zero();
            let one = Field::one();
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

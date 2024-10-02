use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    parser::types::{GenericParameters, TyKind},
    stdlib::bits::to_bits,
    var::{ConstOrCell, Value, Var},
};

use super::boolean;

use ark_ff::{Field, One, PrimeField, Zero};
use kimchi::o1_utils::FieldHelpers;

use std::ops::Neg;

/// Negates a field element
pub fn neg<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cvar: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
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
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    match (lhs, rhs) {
        // 2 constants
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => Var::new_constant(*lhs + *rhs, span),

        // const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            // if the constant is zero, we can ignore this gate
            if cst.is_zero() {
                // TODO: that span is incorrect, it should come from lhs or rhs...
                return Var::new_var(cvar.clone(), span);
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
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    let neg_rhs = neg(compiler, rhs, span);
    add(compiler, lhs, &neg_rhs.cvars[0], span)
}

/// Multiplies two field elements
pub fn mul<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
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

fn constrain_div_mod<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> (B::Var, B::Var) {
    // to constrain lhs − q * rhs − rem = 0
    // where rhs is the modulus
    // so 0 <= rem < rhs

    let one = B::Field::one();

    // todo: to avoid duplicating a lot of code due the different combinations of the input types
    // until we refactor the backend to handle ConstOrCell or some kind of wrapper that encapsulate the different variable types
    // convert cst to var for easier handling
    let lhs = match lhs {
        ConstOrCell::Const(lhs) => compiler.backend.add_constant(
            Some("wrap a constant as var"),
            *lhs,
            span,
        ),
        ConstOrCell::Cell(lhs) => lhs.clone(),
    };

    let rhs = match rhs {
        ConstOrCell::Const(rhs) => compiler.backend.add_constant(
            Some("wrap a constant as var"),
            *rhs,
            span,
        ),
        ConstOrCell::Cell(rhs) => rhs.clone(),
    };

    // witness var for quotient
    let q = Value::VarDivVar(lhs.clone(), rhs.clone());
    let q_var = compiler.backend.new_internal_var(q, span);

    // witness var for remainder
    let rem = Value::VarModVar(lhs.clone(), rhs.clone());
    let rem_var = compiler.backend.new_internal_var(rem, span);

    // rem < rhs
    let lt_rem = &less_than(compiler, None, &ConstOrCell::Cell(rem_var.clone()), &ConstOrCell::Cell(rhs.clone()), span)[0];
    let lt_rem = lt_rem.cvar().expect("expected a cell var");
    compiler.backend.assert_eq_const(lt_rem, one, span);

    // foundamental constraint: lhs - q * rhs - rem = 0
    let q_mul_rhs = compiler.backend.mul(&q_var, &rhs, span);
    let lhs_sub_q_mul_rhs = compiler.backend.sub(&lhs, &q_mul_rhs, span);

    // cell representing the foundamental constraint
    let fc_var = compiler.backend.sub(&lhs_sub_q_mul_rhs, &rem_var, span);
    compiler.backend.assert_eq_const(&fc_var, B::Field::zero(), span);

    (rem_var, q_var)
}

/// Divide operation
pub fn div<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    // to constrain lhs − q * rhs − rem = 0
    // rhs can't be zero
    match rhs {
        ConstOrCell::Const(rhs) => {
            if rhs.is_zero() {
                panic!("division by zero");
            }
        }
        _ => {
            let is_zero = is_zero_cell(compiler, rhs, span);
            let is_zero = is_zero[0].cvar().unwrap();
            compiler.backend.assert_eq_const(is_zero, B::Field::zero(), span);
        }
    };

    match (lhs, rhs) {
        // if rhs is a constant, we can just divide lhs by rhs
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
            // to bigint
            let lhs = lhs.to_biguint();
            let rhs = rhs.to_biguint();
            let res = lhs / rhs;

            Var::new_constant(B::Field::from(res), span)
        }
        _ => {
            let (_, q) = constrain_div_mod(compiler, lhs, rhs, span);
            Var::new_var(q, span)
        },
    }
}

/// Modulus operation
pub fn modulus<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    // to constrain lhs − q * rhs − rem = 0

    let zero = B::Field::zero();

    // rhs can't be zero
    match &rhs {
        ConstOrCell::Const(rhs) => {
            if rhs.is_zero() {
                panic!("modulus by zero");
            }
        }
        _ => {
            let is_zero = is_zero_cell(compiler, rhs, span);
            let is_zero = is_zero[0].cvar().unwrap();
            compiler.backend.assert_eq_const(is_zero, zero, span);
        }
    };

    match (lhs, rhs) {
        // if rhs is a constant, we can just divide lhs by rhs
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
            let lhs = lhs.to_biguint();
            let rhs = rhs.to_biguint();
            let res = lhs % rhs;

            Var::new_constant(res.into(), span)
        }
        _ => {
            let (rem, _) = constrain_div_mod(compiler, lhs, rhs, span);
            Var::new_var(rem, span)
        }
    }
}

/// Left shift operation
pub fn left_shift<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    // to constrain lhs * (1 << rhs) = res

    match (lhs, rhs) {
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
            // convert to bigint
            let pow2 = B::Field::from(2u32).pow(rhs.into_repr());
            let res = *lhs * pow2;
            Var::new_constant(res, span)
        }
        (ConstOrCell::Cell(lhs_), ConstOrCell::Const(rhs_)) => {
            let pow2 = B::Field::from(2u32).pow(rhs_.into_repr());
            let res = compiler.backend.mul_const(lhs_, &pow2, span);
            Var::new_var(res, span)
        }
        // todo: wrap rhs in a symbolic value
        (ConstOrCell::Const(_), ConstOrCell::Cell(_)) => todo!(),
        (ConstOrCell::Cell(_), ConstOrCell::Cell(_)) => todo!(),
    }
}

/// This takes variables that can be anything, and returns a boolean
// TODO: so perhaps it's not really relevant in this file?
pub fn equal<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &Var<B::Field, B::Var>,
    rhs: &Var<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
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
    x1: &ConstOrCell<B::Field, B::Var>,
    x2: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    // These four constraints are enough:
    //
    // 1. `diff = x2 - x1`
    // 2. `diff_inv * diff = one_minus_res`
    // 3. `one_minus_res = 1 - res`
    // 4. `res * diff = 0`
    //
    // To prove this, it suffices to prove that:
    //
    // a. `diff = 0 => res = 1`.
    // b. `diff != 0 => res = 0`.
    //
    // Proof:
    //
    // a. if `diff = 0`,
    //      then using (2) `one_minus_res = 0`,
    //      then using (3) `res = 1`
    //
    // b. if `diff != 0`
    //      then using (4) `res = 0`
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
                ConstOrCell::Cell(cvar) => cvar.clone(),
            };

            let x2 = match x2 {
                ConstOrCell::Const(cst) => compiler.backend.add_constant(
                    Some("encode the rhs constant of the equality check in the circuit"),
                    *cst,
                    span,
                ),
                ConstOrCell::Cell(cvar) => cvar.clone(),
            };

            // 1. diff = x2 - x1
            let diff = compiler.backend.sub(&x2, &x1, span);
            let diff_inv = compiler
                .backend
                .new_internal_var(Value::Inverse(diff.clone()), span);

            // 2. diff_inv * diff = one_minus_res
            let diff_inv_mul_diff = compiler.backend.mul(&diff_inv, &diff, span);

            // 3. one_minus_res = 1 - res
            // => res = 1 - diff_inv * diff
            let res = compiler.backend.new_internal_var(
                Value::LinearCombination(vec![(one.neg(), diff_inv_mul_diff.clone())], one),
                span,
            );
            let neg_res = compiler.backend.neg(&res, span);
            let one_minus_res = compiler.backend.add_const(&neg_res, &one, span);
            compiler
                .backend
                .assert_eq_var(&diff_inv_mul_diff, &one_minus_res, span);

            // 4. res * diff = 0
            let res_mul_diff = compiler.backend.mul(&res, &diff, span);
            compiler.backend.assert_eq_const(&res_mul_diff, zero, span);

            Var::new_var(res, span)
        }
    }
}

/// Returns 1 if lhs != rhs, 0 otherwise
pub fn not_equal<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    lhs: &Var<B::Field, B::Var>,
    rhs: &Var<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    assert_eq!(lhs.len(), rhs.len());

    let one = B::Field::one();

    if lhs.len() == 1 {
        let diff = sub(compiler, &lhs[0], &rhs[0], span);
        let is_zero = is_zero_cell(compiler, &diff[0], span);
        return boolean::not(compiler, &is_zero[0], span);
    }

    let acc = compiler.backend.add_constant(
        Some("start accumulator at 1 for the inequality check"),
        one,
        span,
    );
    let mut acc = Var::new_var(acc, span);

    for (l, r) in lhs.cvars.iter().zip(&rhs.cvars) {
        let diff = sub(compiler, l, r, span);
        let res = is_zero_cell(compiler, &diff[0], span);
        let not_res = boolean::not(compiler, &res[0], span);
        acc = boolean::and(compiler, &not_res[0], &acc[0], span);
    }

    acc
}

/// Returns 1 if lhs < rhs, 0 otherwise
pub fn less_than<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    bitlen: Option<usize>,
    lhs: &ConstOrCell<B::Field, B::Var>,
    rhs: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    let one = B::Field::one();
    let zero = B::Field::zero();

    // Instead of comparing bit by bit, we check the carry bit:
    // lhs + (1 << LEN) - rhs
    // proof:
    // lhs + (1 << LEN) will add a carry bit, valued 1, to the bit array representing lhs,
    // resulted in a bit array of length LEN + 1, named as sum_bits.
    // if `lhs < rhs``, then `lhs - rhs < 0`, thus `(1 << LEN) + lhs - rhs < (1 << LEN)`
    // then, the carry bit of sum_bits is 0.
    // otherwise, the carry bit of sum_bits is 1.

    /*
    psuedo code:
    let carry_bit_len = LEN + 1;

    # 1 << LEN
    let mut pow2 = 1;
    for ii in 0..LEN {
        pow2 = pow2 + pow2;
    }

    let sum = (pow2 + lhs) - rhs;
    let sum_bit = bits::to_bits(carry_bit_len, sum);

    let b1 = false;
    let b2 = true;
    let res = if sum_bit[LEN] { b1 } else { b2 };

    */

    let modulus_bits: usize = B::Field::modulus_biguint()
        .bits()
        .try_into()
        .expect("can't determine the number of bits in the modulus");

    let bitlen_upper_bound = modulus_bits - 2;
    let bit_len = bitlen.unwrap_or(bitlen_upper_bound);

    assert!(bit_len <= (bitlen_upper_bound));


    let carry_bit_len = bit_len + 1;


    // let pow2 = (1 << bit_len) as u32;
    // let pow2 = B::Field::from(pow2);
    let two = B::Field::from(2u32);
    let pow2 = two.pow([bit_len as u64]);

    // let pow2_lhs = compiler.backend.add_const(lhs, &pow2, span);
    match (lhs, rhs) {
        (ConstOrCell::Const(lhs), ConstOrCell::Const(rhs)) => {
            let res = if lhs < rhs { one } else { zero };

            Var::new_constant(res, span)
        }
        (_, _) => {
            let pow2_lhs = match lhs {
                // todo: we really should refactor the backend to handle ConstOrCell
                ConstOrCell::Const(lhs) => compiler.backend.add_constant(
                    Some("wrap a constant as var"),
                    *lhs + pow2,
                    span,
                ),
                ConstOrCell::Cell(lhs) => compiler.backend.add_const(lhs, &pow2, span),
            };

            let rhs = match rhs {
                ConstOrCell::Const(rhs) => compiler.backend.add_constant(
                    Some("wrap a constant as var"),
                    *rhs,
                    span,
                ),
                ConstOrCell::Cell(rhs) => rhs.clone(),
            };

            let sum = compiler.backend.sub(&pow2_lhs, &rhs, span);

            // todo: this api call is kind of weird here, maybe these bulitin shouldn't get inputs from the `GenericParameters`
            let generic_var_name = "LEN".to_string();
            let mut gens = GenericParameters::default();
            gens.add(generic_var_name.clone());
            gens.assign(&generic_var_name, carry_bit_len as u32, span)
                .unwrap();

            // construct var info for sum
            let cbl_var = Var::new_constant(B::Field::from(carry_bit_len as u32), span);
            let cbl_var = VarInfo::new(cbl_var, false, Some(TyKind::Field { constant: true }));

            let sum_var = Var::new_var(sum, span);
            let sum_var = VarInfo::new(sum_var, false, Some(TyKind::Field { constant: false }));

            let sum_bits = to_bits(compiler, &gens, &[cbl_var, sum_var], span).unwrap().unwrap();
            // convert to cell vars
            let sum_bits: Vec<_> = sum_bits.cvars.into_iter().collect();

            // if sum_bits[LEN] == 0, then lhs < rhs
            let res = &is_zero_cell(compiler, &sum_bits[bit_len], span)[0];
            let res = res
                .cvar()
                .unwrap();
            Var::new_var(res.clone(), span)
        }
    }
}

/// Returns 1 if var is zero, 0 otherwise
fn is_zero_cell<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    var: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    let zero = B::Field::zero();
    let one = B::Field::one();

    match var {
        ConstOrCell::Const(a) => {
            let res = if *a == zero { one } else { zero };
            Var::new_constant(res, span)
        }
        ConstOrCell::Cell(a) => {
            // x = 1 / a -- inverse of input
            let x = compiler
                .backend
                .new_internal_var(Value::Inverse(a.clone()), span);

            // m = -a*x + 1 -- constrain m to be 1 if a == 0
            let ax = compiler.backend.mul(&a, &x, span);
            let neg_ax = compiler.backend.neg(&ax, span);
            let m = compiler.backend.new_internal_var(
                Value::LinearCombination(vec![(one, neg_ax.clone())], one),
                span,
            );
            let m_sub_one = compiler.backend.add_const(&m, &one.neg(), span);

            compiler.backend.assert_eq_var(&neg_ax, &m_sub_one, span);

            // a * m = 0 -- constrain m to be 0 if a != 0
            let a_mul_m = compiler.backend.mul(&a, &m, span);

            compiler.backend.assert_eq_const(&a_mul_m, zero, span);

            Var::new_var(m, span)
        }
    }
}

pub fn if_else<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cond: &Var<B::Field, B::Var>,
    then_: &Var<B::Field, B::Var>,
    else_: &Var<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    assert_eq!(cond.len(), 1);
    assert_eq!(then_.len(), else_.len());

    let cond = &cond[0];

    let mut vars = vec![];

    for (then_, else_) in then_.cvars.iter().zip(&else_.cvars) {
        let var = if_else_inner(compiler, cond, then_, else_, span);
        vars.push(var[0].clone());
    }

    Var::new(vars, span)
}

pub fn if_else_inner<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    cond: &ConstOrCell<B::Field, B::Var>,
    then_: &ConstOrCell<B::Field, B::Var>,
    else_: &ConstOrCell<B::Field, B::Var>,
    span: Span,
) -> Var<B::Field, B::Var> {
    // we need to constrain:
    //
    // * res = (1 - cond) * else + cond * then
    //

    // if cond is constant, easy
    if let ConstOrCell::Const(cond) = cond {
        if cond.is_one() {
            return Var::new_cvar(then_.clone(), span);
        } else {
            return Var::new_cvar(else_.clone(), span);
        }
    }

    // determine the result via arithemtic
    let cond_then = mul(compiler, then_, cond, span);
    let one = ConstOrCell::Const(B::Field::one());
    let one_minus_cond = sub(compiler, &one, cond, span);
    let temp = mul(compiler, &one_minus_cond[0], else_, span);
    add(compiler, &cond_then[0], &temp[0], span)
}

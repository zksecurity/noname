use crate::error;

use super::test_stdlib;
use ark_ff::One;
use error::Result;
use num_bigint::BigInt;

#[test]
fn test_module_sum_with_carry() -> Result<()> {
    let public_inputs = r#"{"lhs": "4"}"#;
    let private_inputs = r#"{"rhs": "5"}"#;

    test_stdlib(
        "bigint/module_sum/main.no",
        "bigint/module_sum/main.asm",
        public_inputs,
        private_inputs,
        vec!["1", "1"],
    )?;

    Ok(())
}

#[test]
fn test_module_sum_without_carry() -> Result<()> {
    let public_inputs = r#"{"lhs": "4"}"#;
    let private_inputs = r#"{"rhs": "3"}"#;

    test_stdlib(
        "bigint/module_sum/main.no",
        "bigint/module_sum/main.asm",
        public_inputs,
        private_inputs,
        vec!["7", "0"],
    )?;

    Ok(())
}

#[test]
fn test_module_sum_three_with_carry() -> Result<()> {
    let public_inputs = r#"{"lhs": "4"}"#;
    let private_inputs = r#"{"rhs": "2"}"#;

    test_stdlib(
        "bigint/module_sum_three/main.no",
        "bigint/module_sum_three/main.asm",
        public_inputs,
        private_inputs,
        vec!["1", "1"],
    )?;

    Ok(())
}

#[test]
fn test_module_sum_three_with_carry_two() -> Result<()> {
    let public_inputs = r#"{"lhs": "7"}"#;
    let private_inputs = r#"{"rhs": "7"}"#;

    test_stdlib(
        "bigint/module_sum_three/main.no",
        "bigint/module_sum_three/main.asm",
        public_inputs,
        private_inputs,
        // 2 carry bits
        vec!["1", "2"],
    )?;

    Ok(())
}

#[test]
fn test_module_sum_three_without_carry() -> Result<()> {
    let public_inputs = r#"{"lhs": "1"}"#;
    let private_inputs = r#"{"rhs": "1"}"#;

    test_stdlib(
        "bigint/module_sum_three/main.no",
        "bigint/module_sum_three/main.asm",
        public_inputs,
        private_inputs,
        vec!["5", "0"],
    )?;

    Ok(())
}

fn bigint_to_array(n: u32, k: u32, x: BigInt) -> Vec<BigInt> {
    // Compute modulus as 2^n
    let modulus = BigInt::one() << n;

    let mut ret = Vec::new();
    let mut x_temp = x;

    for _ in 0..k {
        // Get the remainder of x_temp divided by modulus
        let remainder = &x_temp % &modulus;
        ret.push(remainder.clone());

        // Divide x_temp by modulus (equivalent to right-shifting by n bits)
        x_temp >>= n;
    }

    ret
}

#[test]
fn test_add_limbs() -> Result<()> {
    let a = bigint_to_array(8, 3, BigInt::from(16));
    let b = bigint_to_array(8, 3, BigInt::from(17));
    let c = bigint_to_array(8, 4, BigInt::from(33));

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let sum_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    // Serialize to JSON string
    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "bigint/add_limbs/main.no",
        "bigint/add_limbs/main.asm",
        &public_inputs,
        &private_inputs,
        sum_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

#[test]
fn test_mult_limbs() -> Result<()> {
    let a = bigint_to_array(8, 3, BigInt::from(10));
    let b = bigint_to_array(8, 3, BigInt::from(10));
    let c = bigint_to_array(8, 6, BigInt::from(100));

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let res_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    // Serialize to JSON string
    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "bigint/mult_limbs/main.no",
        "bigint/mult_limbs/main.asm",
        &public_inputs,
        &private_inputs,
        res_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

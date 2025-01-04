use crate::error;

use super::test_stdlib;
use ark_ff::One;
use error::Result;
use num_bigint::BigInt;
use rstest::rstest;

#[test]
fn test_module_sum_with_carry() -> Result<()> {
    let public_inputs = r#"{"lhs": "4"}"#;
    let private_inputs = r#"{"rhs": "5"}"#;

    test_stdlib(
        "biguint/module_sum/main.no",
        Some("biguint/module_sum/main.asm"),
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
        "biguint/module_sum/main.no",
        Some("biguint/module_sum/main.asm"),
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
        "biguint/module_sum_three/main.no",
        Some("biguint/module_sum_three/main.asm"),
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
        "biguint/module_sum_three/main.no",
        Some("biguint/module_sum_three/main.asm"),
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
        "biguint/module_sum_three/main.no",
        Some("biguint/module_sum_three/main.asm"),
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
        "biguint/add_limbs/main.no",
        Some("biguint/add_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        sum_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

#[test]
fn test_mult_limbs() -> Result<()> {
    let a = bigint_to_array(8, 3, BigInt::from(16777215));
    let b = bigint_to_array(8, 3, BigInt::from(16777215));
    let c = bigint_to_array(8, 6, BigInt::from(281474943156225u64));

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let res_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    // Serialize to JSON string
    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "biguint/mult_limbs/main.no",
        Some("biguint/mult_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        res_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

#[rstest]
#[case(BigInt::from(5), BigInt::from(5), BigInt::from(1))]
#[case(BigInt::from(9), BigInt::from(5), BigInt::from(1))]
#[case(BigInt::from(10), BigInt::from(5), BigInt::from(2))]
#[case(BigInt::from(13), BigInt::from(5), BigInt::from(2))]
// #[case(BigInt::from(50), BigInt::from(5), BigInt::from(10))]
fn test_div_limbs(
    #[case] lhs_val: BigInt,
    #[case] rhs_val: BigInt,
    #[case] res_val: BigInt,
) -> Result<()> {
    let a = bigint_to_array(2, 3, lhs_val);
    let b = bigint_to_array(2, 2, rhs_val);
    let c = bigint_to_array(2, 2, res_val);

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let res_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "biguint/div_limbs/main.no",
        Some("biguint/div_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        res_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

#[rstest]
#[case(BigInt::from(5), BigInt::from(5), BigInt::from(0))]
#[case(BigInt::from(9), BigInt::from(5), BigInt::from(4))]
#[case(BigInt::from(13), BigInt::from(5), BigInt::from(3))]
fn test_rem_limbs(
    #[case] lhs_val: BigInt,
    #[case] rhs_val: BigInt,
    #[case] res_val: BigInt,
) -> Result<()> {
    let a = bigint_to_array(2, 3, lhs_val);
    let b = bigint_to_array(2, 2, rhs_val);
    let c = bigint_to_array(2, 2, res_val);

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let res_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "biguint/rem_limbs/main.no",
        Some("biguint/rem_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        res_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

#[rstest]
#[case(BigInt::from(5), BigInt::from(5), 0_u8)]
#[case(BigInt::from(9), BigInt::from(5), 0_u8)]
#[case(BigInt::from(2), BigInt::from(5), 1_u8)]
fn test_less_than_limbs(
    #[case] lhs_val: BigInt,
    #[case] rhs_val: BigInt,
    #[case] res_val: u8,
) -> Result<()> {
    let a = bigint_to_array(2, 3, lhs_val);
    let b = bigint_to_array(2, 3, rhs_val);

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();

    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "biguint/less_than_limbs/main.no",
        Some("biguint/less_than_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        vec![res_val.to_string().as_str()],
    )?;

    Ok(())
}

#[rstest]
#[case(BigInt::from(6), BigInt::from(5), BigInt::from(1))]
#[case(BigInt::from(12), BigInt::from(5), BigInt::from(2))]
#[case(BigInt::from(5), BigInt::from(5), BigInt::from(0))]
fn test_mod_limbs(
    #[case] lhs_val: BigInt,
    #[case] rhs_val: BigInt,
    #[case] res_val: BigInt,
) -> Result<()> {
    let a = bigint_to_array(2, 3, lhs_val);
    let b = bigint_to_array(2, 2, rhs_val);
    let c = bigint_to_array(2, 2, res_val);

    let lhs_strings: Vec<String> = a.iter().map(|num| num.to_string()).collect();
    let rhs_strings: Vec<String> = b.iter().map(|num| num.to_string()).collect();
    let res_strings: Vec<String> = c.iter().map(|num| num.to_string()).collect();

    let lhs_strings = serde_json::to_string(&lhs_strings).unwrap();
    let rhs_strings = serde_json::to_string(&rhs_strings).unwrap();

    let public_inputs = format!(r#"{{"lhs": {}}}"#, lhs_strings);
    let private_inputs = format!(r#"{{"rhs": {}}}"#, rhs_strings);

    test_stdlib(
        "biguint/mod_limbs/main.no",
        Some("biguint/mod_limbs/main.asm"),
        &public_inputs,
        &private_inputs,
        res_strings.iter().map(|s| s.as_str()).collect(),
    )?;

    Ok(())
}

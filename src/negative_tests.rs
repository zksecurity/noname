use crate::{
    backends::kimchi::KimchiVesta, compiler::{typecheck_next_file_inner, Sources}, error::{ErrorKind, Result}, mast::Mast, type_checker::TypeChecker
};

fn tast_pass(code: &str) -> (Result<usize>, TypeChecker<KimchiVesta>) {
    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    (res, tast)
}

fn mast_pass(code: &str) -> Result<Mast<KimchiVesta>> {
    let tast = tast_pass(code).1;
    crate::mast::monomorphize(tast)
}

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::NoReturnExpected));
}

#[test]
fn test_return_expected() {
    // return expected
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::MissingReturn));
}

#[test]
fn test_return_mismatch() {
    // return type mismatch
    let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}

#[test]
fn test_generic_const_for_loop() {
    let code = r#"
        // generic on const argument
        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }

        fn loop() {
            for ii in 0..3 {
                gen(ii);
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::GenericInForLoop));
}

#[test]
fn test_generic_array_for_loop() {
    let code = r#"
        // generic on array argument
        fn gen(arr: [Field; LEN]) -> [Field; LEN] {
            return arr;
        }

        fn loop() {
            let arr = [0; 3];
            for ii in 0..3 {
                gen([0; ii]);
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::GenericInForLoop));
}

#[test]
fn test_generic_missing_parameter_arg() {
    let code = r#"
        fn gen(const len: Field) -> [Field; LEN] {
            return [0; LEN];
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::UndefinedVariable
    ));
}

#[test]
fn test_generic_ret_type_mismatched() {
    let code = r#"
        // mast phase should catch the type mismatch in the return type
        fn gen(const LEN: Field) -> [Field; 2] {
            return [0; LEN];
        }
        fn main(pub xx: Field) {
            let ret = gen(3);
        }
        "#;

    let res = mast_pass(code).err();
    assert!(matches!(
        res.unwrap().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}
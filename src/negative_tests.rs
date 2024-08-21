use crate::{
    backends::kimchi::KimchiVesta,
    compiler::{typecheck_next_file_inner, Sources},
    error::ErrorKind,
    type_checker::TypeChecker,
};

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

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

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

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

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

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

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

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

    let mut tast = TypeChecker::<KimchiVesta>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut Sources::new(),
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::GenericInForLoop));
}

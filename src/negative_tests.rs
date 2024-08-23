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
fn test_const_attr_mismatch() {
    let code = r#"
        struct House {
            room_num: [Field; 2],
        }

        fn House.room(self, const idx: Field) -> Field {
            return self.room_num[idx];
        }

        fn main(pub xx: Field) -> Field {
            let house = House { room_num: [1, 2] };

            // xx is not a constant
            return house.room(xx);
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
        ErrorKind::ArgumentTypeMismatch(..)
    ));
}

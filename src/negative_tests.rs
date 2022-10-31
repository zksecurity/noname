use crate::{compiler::get_tast_single, error::ErrorKind};

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let res = get_tast_single("example.no".to_string(), code.to_string());

    assert!(matches!(res.unwrap_err().kind, ErrorKind::NoReturnExpected));

    // return expected
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    }
    "#;

    let res = get_tast_single("example.no".to_string(), code.to_string());

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MissingReturn));

    // return type mismatch
    let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

    let res = get_tast_single("example.no".to_string(), code.to_string());

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}

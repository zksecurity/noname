use crate::{
    compiler::{get_tast_inner, Sources},
    error::ErrorKind,
    type_checker::Dependencies,
};

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let mut sources = Sources::new();
    let deps = &Dependencies::default();

    let res = get_tast_inner(
        &mut sources,
        "example.no".to_string(),
        code.to_string(),
        deps,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::NoReturnExpected));

    // return expected
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    }
    "#;

    let res = get_tast_inner(
        &mut sources,
        "example.no".to_string(),
        code.to_string(),
        deps,
    );

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MissingReturn));

    // return type mismatch
    let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

    let res = get_tast_inner(
        &mut sources,
        "example.no".to_string(),
        code.to_string(),
        deps,
    );

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}

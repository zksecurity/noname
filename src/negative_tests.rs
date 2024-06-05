macro_rules! create_test_with_error {
    ($test_name: ident, $expected_error: pat, $code: expr) => {
        #[test]
        fn $test_name() {
            let mut tast = TypeChecker::<KimchiVesta>::new();
            let res = typecheck_next_file_inner(
                &mut tast,
                None,
                &mut Sources::new(),
                "example.no".to_string(),
                $code.to_string(),
                0,
            );
            let actual_error = res.unwrap_err().kind;

            assert!(matches!(actual_error, $expected_error));

        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        backends::kimchi::KimchiVesta,
        compiler::{typecheck_next_file_inner, Sources},
        error::ErrorKind,
        type_checker::TypeChecker,
    };

    create_test_with_error!(
        test_return,
        ErrorKind::NoReturnExpected,
        r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#
    );

    create_test_with_error!(
        test_return_expected,
        ErrorKind::MissingReturn,
        r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    } 
    "#
    );

    create_test_with_error!(
        return_mismatch,
        ErrorKind::ReturnTypeMismatch(..),
        r#"
    fn thing(xx: Field) -> Field {
        return true;
    } 
    "#
    );

    create_test_with_error!(
        duplicate_definition,
        ErrorKind::DuplicateDefinition(_),
        r#"
    fn thing(xx: Field) -> Field {
        let xx = 10;
        let xx = 12;
    } 
    "#
    );
}

#[cfg(test)]
mod tests {
    use crate::{
        backends::kimchi::KimchiVesta,
        compiler::{typecheck_next_file_inner, Sources},
        error::ErrorKind,
        type_checker::TypeChecker,
    };

    fn run_type_check_for_invalid_code(code: &str) -> crate::error::Result<usize> {
        let mut tast = TypeChecker::<KimchiVesta>::new();
        typecheck_next_file_inner(
            &mut tast,
            None,
            &mut Sources::new(),
            "example.no".to_string(),
            code.to_string(),
            0,
        )
    }

    #[test]
    #[should_panic(expected = "NoReturnExpected")]
    fn test_return() {
        // no return expected
        let code = r#"
        fn thing(xx: Field) {
            return xx;
        }
        "#;
        run_type_check_for_invalid_code(code).unwrap();
    }

    #[test]
    #[should_panic(expected = "MissingReturn")]
    fn test_return_expected() {
        // return expected
        let code = r#"
        fn thing(xx: Field) -> Field {
            let yy = xx + 1;
        }
        "#;

        run_type_check_for_invalid_code(code).unwrap();
    }

    #[test]
    #[should_panic(expected = "ReturnTypeMismatch(Field, Bool)")]
    fn test_return_mismatch() {
        // return type mismatch
        let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

        run_type_check_for_invalid_code(code).unwrap();
    }

    #[test]
    #[should_panic(expected = "DuplicateDefinition(\"xx\")")]
    fn test_duplicate_definition() {
        let code = r#"
    fn thing(xx: Field) -> Field {
        let xx = 10;
        let xx = 12;
    } 
    "#;

        run_type_check_for_invalid_code(code).unwrap();
    }
}

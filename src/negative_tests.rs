//
// Type checker tests
//

use crate::{
    error::{ErrorKind, Result},
    lexer::Token,
    parser::AST,
    type_checker::{Dependencies, TAST},
};

fn type_check(code: &str) -> Result<TAST> {
    // lexer
    let tokens = Token::parse(code).unwrap();
    // AST
    let ast = AST::parse(tokens).unwrap();
    // TAST
    let deps = Dependencies::default();
    TAST::analyze(ast, &deps)
}

#[test]
fn test_return() {
    // no return expected
    let code = r#"
    fn thing(xx: Field) {
        return xx;
    }
    "#;

    let res = type_check(&code);

    assert!(matches!(res.unwrap_err().kind, ErrorKind::NoReturnExpected));

    // return expected
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
    }
    "#;

    let res = type_check(&code);

    assert!(matches!(res.unwrap_err().kind, ErrorKind::MissingReturn));

    // return type mismatch
    let code = r#"
        fn thing(xx: Field) -> Field {
            return true;
        }
        "#;

    let res = type_check(&code);

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ReturnTypeMismatch(..)
    ));
}

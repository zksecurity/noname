use crate::{
    backends::r1cs::{R1csBn254Field, R1CS},
    circuit_writer::CircuitWriter,
    compiler::{get_nast, typecheck_next_file_inner, Sources},
    error::{ErrorKind, Result},
    mast::Mast,
    name_resolution::NAST,
    type_checker::TypeChecker,
    witness::CompiledCircuit,
};

fn nast_pass(code: &str) -> Result<(NAST<R1CS<R1csBn254Field>>, usize)> {
    let mut source = Sources::new();
    let res = get_nast(
        None,
        &mut source,
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    res
}

fn tast_pass(code: &str) -> (Result<usize>, TypeChecker<R1CS<R1csBn254Field>>, Sources) {
    let mut source = Sources::new();
    let mut tast = TypeChecker::<R1CS<R1csBn254Field>>::new();
    let res = typecheck_next_file_inner(
        &mut tast,
        None,
        &mut source,
        "example.no".to_string(),
        code.to_string(),
        0,
    );

    (res, tast, source)
}

fn mast_pass(code: &str) -> Result<Mast<R1CS<R1csBn254Field>>> {
    let (_, tast, _) = tast_pass(code);
    crate::mast::monomorphize(tast)
}

fn synthesizer_pass(code: &str) -> Result<CompiledCircuit<R1CS<R1csBn254Field>>> {
    let mast = mast_pass(code);
    CircuitWriter::generate_circuit(mast?, R1CS::new())
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

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
}

#[test]
fn test_generic_const_nested_for_loop() {
    let code = r#"
        // generic on const argument
        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }

        fn loop() {
            let mut arr = [0; 3];
            for ii in 0..3 {
                for jj in 0..3 {
                    gen(ii);
                }
            }
        }
        "#;

    let res = tast_pass(code).0;
    println!("{:?}", res);

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
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
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
}

#[test]
fn test_generic_method_cst_for_loop() {
    let code = r#"
        struct Thing {
            xx: Field,
        }

        // generic on const argument
        fn Thing.gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }

        fn loop() {
            let thing = Thing { xx: 3 };
            for ii in 0..3 {
                thing.gen(ii);
            }
        }
        "#;

    let res = tast_pass(code).0;

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
}

#[test]
fn test_generic_method_array_for_loop() {
    let code = r#"
        struct Thing {
            xx: Field,
        }

        // generic on array argument
        fn Thing.gen(arr: [Field; LEN]) -> [Field; LEN] {
            return arr;
        }

        fn loop() {
            let thing = Thing { xx: 3 };
            for ii in 0..3 {
                thing.gen([0; ii]);
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
}

#[test]
fn test_generic_method_nested_for_loop() {
    let code = r#"
        struct Thing {
            xx: Field,
        }

        // generic on array argument
        fn Thing.gen(arr: [Field; LEN]) -> [Field; LEN] {
            return arr;
        }

        fn loop() {
            let thing = Thing { xx: 3 };
            for ii in 0..3 {
                for jj in 0..3 {
                    thing.gen([0; ii]);
                }
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::VarAccessForbiddenInForLoop(..)
    ));
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
fn test_generic_symbolic_size_mismatched() {
    let code = r#"
        fn gen(const LEN: Field) -> [Field; 2] {
            return [0; LEN];
        }

        fn main(pub xx: Field) {
            gen(3);
        }
        "#;

    let res = mast_pass(code).err();

    assert!(matches!(
        res.unwrap().kind,
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

    let res = tast_pass(code).0;

    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::ArgumentTypeMismatch(..)
    ));
}

#[test]
fn test_generic_type_mismatched() {
    let code = r#"
        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }
        
        fn comp(arr1: [Field; LEN], arr2: [Field; LEN]) {
            for ii in 0..LEN {
                assert_eq(arr1[ii], arr2[ii]);
            }
        }

        fn main(pub xx: Field) {
            let arr1 = gen(2);
            let arr2 = gen(3);
            comp(arr1, arr2);
        }
        "#;

    let res = mast_pass(code).err();
    assert!(matches!(
        res.unwrap().kind,
        ErrorKind::ConflictGenericValue(..)
    ));
}

#[test]
fn test_generic_assignment_type_mismatched() {
    let code = r#"
        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }
        
        fn main(pub xx: Field) {
            let mut arr = [0; 3];
            arr = gen(2);
        }
        "#;

    let res = mast_pass(code).err();
    assert!(matches!(
        res.unwrap().kind,
        ErrorKind::AssignmentTypeMismatch(..)
    ));
}

#[test]
fn test_generic_custom_type_mismatched() {
    let code = r#"
        struct Thing {
            xx: [Field; 2],
        }

        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }
        
        fn main(pub xx: Field) {
            let arr = gen(3);
            let thing = Thing { xx: arr };
        }
        "#;

    let res = mast_pass(code).err();
    assert!(matches!(
        res.unwrap().kind,
        ErrorKind::InvalidStructFieldType(..)
    ));
}

#[test]
fn test_array_bounds() {
    let code = r#"
        fn gen(const LEN: Field) -> [Field; LEN] {
            return [0; LEN];
        }
        fn main(pub xx: Field) {
            let arr = gen(3);
            arr[3] = 1;
        }
        "#;

    let res = synthesizer_pass(code).err();
    assert!(matches!(
        res.unwrap().kind,
        ErrorKind::ArrayIndexOutOfBounds(..)
    ));
}

#[test]
fn test_iterator_type() {
    let code: &str = r#"
        fn main(pub xx: Field) {
            for ii in xx {
                assert_eq(ii, 3);
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::InvalidIteratorType(..)
    ));
}

#[test]
fn test_iterator_variable_immutable() {
    let code: &str = r#"
        fn main(pub arr: [Field; 3]) {
            for elem in arr {
                elem = 1;
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::AssignmentToImmutableVariable
    ));
}

#[test]
fn test_iterator_variable_redefinition() {
    let code: &str = r#"
        fn main(pub arr: [Field; 3]) {
            let mut var = 5;
            for var in arr {
                assert_eq(var, 1);
            }
        }
        "#;

    let res = tast_pass(code).0;
    assert!(matches!(
        res.unwrap_err().kind,
        ErrorKind::DuplicateDefinition(..)
    ));
}

#[test]
fn test_boolean_and_fail() {
    let code = r#"
    fn thing(xx: Field, yy: Bool) {
        let zz = xx && yy;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

#[test]
fn test_boolean_or_fail() {
    let code = r#"
    fn thing(xx: Field, yy: Bool) {
        let zz = xx || yy;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

#[test]
fn test_addition_mismatch() {
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx + 1;
        return yy + true;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

#[test]
fn test_multiplication_mismatch() {
    let code = r#"
    fn thing(xx: Field) -> Field {
        let yy = xx * 2;
        return yy + true;
    }
    "#;

    let res = tast_pass(code).0;
    assert!(matches!(res.unwrap_err().kind, ErrorKind::MismatchType(..)));
}

#[test]
fn test_generic_missing_parenthesis() {
    let code = r#"
    fn init_arr(const LEFT: Field) -> [Field; 1 + LEFT * 2] {
        let arr = [0; 1 + (LEFT * 2)];
        return arr;
    }
    "#;

    let res = nast_pass(code).err();
    println!("{:?}", res);
    assert!(matches!(res.unwrap().kind, ErrorKind::MissingParenthesis));
}

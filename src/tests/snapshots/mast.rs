use crate::negative_tests::mast_pass;

fn test_monomorphization(code_snippet: &str) {
    let mast = mast_pass(code_snippet).expect("Mast pass failed!");

    // Test functions
    for (fully_qualified, fn_info) in &mast.0.functions {
        insta::assert_snapshot!(serde_json::to_string_pretty(fully_qualified).unwrap());
        insta::assert_snapshot!(serde_json::to_string_pretty(fn_info).unwrap());
    }

    // Test node types
    for (counter, ty_kind) in &mast.0.node_types {
        insta::assert_snapshot!(serde_json::to_string_pretty(counter).unwrap());
        insta::assert_snapshot!(serde_json::to_string_pretty(ty_kind).unwrap());
    }

    // Test structs/constants/node_id
    insta::assert_snapshot!(serde_json::to_string_pretty(&mast.0.structs).unwrap());
    insta::assert_snapshot!(serde_json::to_string_pretty(&mast.0.constants).unwrap());
    insta::assert_snapshot!(serde_json::to_string_pretty(&mast.0.node_id).unwrap());
}

#[test]
fn test_generic_array_access() {
    let code = r#"
    fn last(arr: [Field; LEN]) -> Field {
        return arr[LEN - 1];
    }
    fn main(pub xx: Field) {
        let array1 = [xx + 1, xx + 2, xx + 3];
        let elm1 = last(array1);
        assert_eq(elm1, 4);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_array_nested() {
    let code = r#"
    fn last(arr: [[Field; NN]; 3]) -> Field {
        let mut newarr = [0; NN * 3];
    
        let mut index = 0;
        for ii in 0..3 {
            let inner = arr[ii];
            for jj in 0..NN {
                newarr[(ii * NN) + jj] = inner[jj];
            }
        }
    
        return newarr[(NN * 3) - 1];
    }
    
    fn main(pub xx: Field) {
        let array = [[xx + 1, xx + 2, xx + 3]; 3];
        let elm = last(array);
        assert_eq(elm, 4);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_builtin_bits() {
    let code = r#"
    use std::bits;

    fn main(pub xx: Field) {
        // calculate on a cell var
        let bits = bits::to_bits(3, xx);
        assert(!bits[0]);
        assert(bits[1]);
        assert(!bits[2]);

        let val = bits::from_bits(bits);
        assert_eq(val, xx);

        // calculate on a constant
        let cst_bits = bits::to_bits(3, 2);
        assert(!cst_bits[0]);
        assert(cst_bits[1]);
        assert(!cst_bits[2]);

        let cst = bits::from_bits(cst_bits);
        assert_eq(cst, xx);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_fn_multi_init() {
    let code = r#"
    fn last(arr: [Field; LEN]) -> Field {
        return arr[LEN - 1];
    }
    
    fn main(pub xx: Field) {
        let arr1 = [xx + 1, xx + 2];
        let arr2 = [xx + 4, xx + 5];
        let e1 = last(arr1); // 3
        let e2 = last(arr2); // 6
        assert_eq(e1 + e2, 9);
    }
    "#;
    test_monomorphization(code);
}

// Optimized test for generic_for_loop.rs
#[test]
fn test_generic_for_loops() {
    let code = r#"
    struct Thing {
        xx: Field,
    }
    
    fn Thing.clone(self, const LEN: Field) -> [Field; LEN] {
        let mut arr = [0; LEN];
        for idx in 0..LEN {
            arr[idx] = self.xx;
        }
        return arr;
    }
    
    fn join(const LEN: Field, arr1: [Field; LLEN], arr2: [Field; RLEN]) -> [Field; LEN] {
        let mut arr = [0; LEN];
        for ii in 0..LLEN {
            arr[ii] = arr1[ii];
        }
    
        for jj in 0..RLEN {
            arr[jj + LLEN] = arr2[jj];
        }
    
        return arr;
    }
    
    fn clone(const LEN: Field, val: Field) -> [Field; LEN] {
        let mut arr = [0; LEN];
        for idx in 0..LEN {
            arr[idx] = val;
        }
        return arr;
    }
    
    fn accumulate_mut(const INIT: Field) -> Field {
        // it shouldn't fold these variables, even they hold constant values
        // it should only fold the generic vars
        let mut zz = INIT;
        for ii in 0..3 {
            zz = zz + zz;
        }
        return zz;
    }
    
    fn main(pub xx: Field) {
        let arr1 = [xx + 1, xx + 2];
        let arr2 = [xx + 3, xx + 4];
        
        let arr = join(4, arr1, arr2);
    
        assert_eq(arr[0], arr1[0]);
        assert_eq(arr[1], arr1[1]);
        assert_eq(arr[2], arr2[0]);
        assert_eq(arr[3], arr2[1]);
    
        // test that the generic function is callable within a for loop
        let mut arr3 = [[0; 2]; 2];
        for idx in 0..2 {
            let cloned = clone(2, idx);
            arr3[idx] = cloned;
        }
    
        // test that the generic method is callable within a for loop
        let thing = Thing { xx: 5 };
        for idx in 0..2 {
            let cloned = thing.clone(2);
            arr3[idx] = cloned;
        }
    
        let init_val = 1;
        let res = accumulate_mut(init_val);
        assert_eq(res, 8);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_iterator() {
    let code = r#"
    fn zip(a1: [Field; LEN], a2: [Field; LEN]) -> [[Field; 2]; LEN] {
        let mut result = [[0; 2]; LEN];
        for index in 0..LEN {
            result[index] = [a1[index], a2[index]];
        }
        return result;
    }
    
    fn main(pub arr: [Field; 3]) -> Field {
        let expected = [1, 2, 3];
        for pair in zip(arr, expected) {
            assert_eq(pair[0], pair[1]);
        }
    
        return arr[0];
    }    
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_method_multi_init() {
    let code = r#"
    struct Thing {
        xx: Field,
    }
    
    fn Thing.gen(self, const LEN: Field) -> [Field; LEN] {
        return [self.xx; LEN];
    }
    
    fn main(pub xx: Field, yy: Field) {
        let thing1 = Thing { xx: xx };
        let arr1 = thing1.gen(2);
        let arr2 = thing1.gen(3); 
        assert_eq(arr1[1], arr2[2]);
    
        let thing2 = Thing { xx: yy };
        let arr3 = thing2.gen(2);
        assert_eq(arr3[1], arr2[2] + 1);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_nested_func() {
    let code = r#"
    fn nested_func(const LEN: Field) -> [Field; LEN] {
        return [0; LEN];
    }
    fn mod_arr(val: Field) -> [Field; 3] {
        let mut result = nested_func(3);
        for idx in 0..3 {
            result[idx] = val;
        }
        return result;
    }
    fn main(pub val: Field) {
        let result = mod_arr(val);
        assert_eq(result[0], val);
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_nested_method() {
    let code = r#"
    struct Thing {
        xx: Field,
    }
    
    fn Thing.nested_func(const LEN: Field) -> [Field; LEN] {
        return [0; LEN];
    }
    
    fn Thing.mod_arr(self) -> [Field; 3] {
        // this generic function should be instantiated
        let mut result = self.nested_func(3);
        for idx in 0..3 {
            result[idx] = self.xx;
        }
        return result;
    }
    
    fn main(pub val: Field) -> [Field; 3] {
        let thing = Thing {xx: val};
        return thing.mod_arr();
    }
    "#;
    test_monomorphization(code);
}

#[test]
fn test_generic_repeated_array() {
    let code = r#"
    fn init_arr(const LEFT: Field) -> [Field; 1 + (LEFT * 2)] {
        let arr = [0; 1 + (LEFT * 2)];
        return arr;
    }
    
    fn main(pub public_input: Field) -> [Field; 3] {
        let mut arr = init_arr(1);
        for ii in 0..3 {
            arr[ii] = public_input;
        }
        return arr;
    }
    "#;
    test_monomorphization(code);
}

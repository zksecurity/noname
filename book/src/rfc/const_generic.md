## Summary
Enable const generic to improve modularization and reusability of the code.

## Motivation
In the current implementation, array initialization requires explicitly stating all element values, which becomes impractical for large arrays. Furthermore, the lack of support for constant variables as array sizes limits the ability to write functions that generically handle arrays of varying sizes. This also necessitates hardcoding loop ranges, reducing flexibility.

Enabling const generic will allow define array sizes with constant variables. This would enhance function reusability and significantly improve modularization.

## Detailed design

### Declare a const argument
Enable functions to define a constant argument, which can be used to define arrays with a default element or as an array size of the return type:

```rust
fn const_generic(const cst: Field) -> [Field; cst] {
    let mut xx = [1; cst];
    for ii in 1..cst {
        xx[ii] = 2;
    }
    return xx;
}
```

### Applying a const as a argument
Use an existing constant to set array size directly in function parameters:

```rust
const cst = 300;
fn const_generic(yy: Field) -> [Field; cst] {
    let xx = [1; cst];
    return xx;
}
```

### Builtin functions

Allow builtin functions to be defined with const arguments:

```rust
use std::to_bits;
use std::from_bits;

const num_bits = 8;

fn main() {
    let val1 = 101;
    let bits = to_bits(num_bits, val); // bits is array ComputedExpr with len
    let val2 = from_bits(bits); // from_bits builtin impl knows the size of the array
    assert_eq!(val1, val2);
}
```

The `to_bits` builtin function accepts a const variable to determine the size of the return array.

```rust
fn to_bits(const num_bits: Field, val: Field) -> [Field; num_bits]
```

The from_bits builtin function can determine the number of bits to convert based on the information that the `ComputedExprKind::Array` embeds in the `bits` variable. We will explain on `ComputedExpr` at the end to show how to improve the structure of the inputs and outputs of these builtin functions.

## Type checking const variable

The existing type checker is designed to do local type checks. For the cases of const variable as array size, the type checker has to trace the values globally. This is out of the scope for the design of the current type checker.

Instead of checking the const variable in the type checker, we can check it in the circuit generation phase. 

### Review how VarOrRef works
 
`VarOrRef` is responsible for containing the underlying variables to pass around during circuit generation.

For example:
```rust
houses[1].rooms[2].size
```

The variable `houses` holds all the `cvars` to represent the underlying low level variables. If `houses` is a mutable variable, it will be a reference, which records the following instead of holding the actual `cvars` array:
 - the variable name `houses` in local scope
 - start index of the `cvars` to represent the current access path
 - the length of the `cvars` to represent size of the element targeted by the current access path

So when it wants to update the elements inside the `houses`, it will need to reference the start index and the length of the `cvars` of the elements, so that it can narrow down the scope of the `cvars` to be updated.

The following code is how it narrows down the scope for a field access or array access:

```rust
pub(crate) fn narrow(&self, start: usize, len: usize) -> Self {
    match self {
        VarOrRef::Var(var) => {
            let cvars = var.range(start, len).to_vec();
            VarOrRef::Var(Var::new(cvars, var.span))
        }

        //      old_start
        //      |
        //      v
        // |----[-----------]-----| <-- var.cvars
        //       <--------->
        //         old_len
        //
        //
        //          start
        //          |
        //          v
        //      |---[-----]-|
        //           <--->
        //            len
        //
        VarOrRef::Ref {
            var_name,
            start: old_start,
            len: old_len,
        } => {
            // ensure that the new range is contained in the older range
            assert!(start < *old_len); // lower bound
            assert!(start + len <= *old_len); // upper bound
            assert!(len > 0); // empty range not allowed

            Self::Ref {
                var_name: var_name.clone(),
                start: old_start + start,
                len,
            }
        }
    }
}
```

This allows to find the range of the `cvars` belonging to the element targeted by the current access path under the local variable behind the `var_name`, which in this case is the `houses`. 

But it doesn't keep track of the structure of the data. For example, when we are accessing `houses[1].rooms[2]`, it should be able to check if the access is within the array bounds. Although the current design can check if it is out of bound by checking access index against the total number of `cvars` for `houses`, it can't label which element access is out of bound as it could be either out of bound of `houses[1]` or the nested access `rooms[2]`.

### Introduce ComputedExpr

With the const variable as array size, we need to propagate its value during circuit generation to check if the access is within the bounds of the array. So we need a way to retain structure information of a variable that is currently represented by `VarOrRef`.

We can add a new layer to the `VarOrRef` to keep track of the structure of the data. It is called `ComputedExpr`, as it is computed from `compute_expr` function, which resolves and passing around the values (`cvars`) via the `Expr`. The key difference from `VarOrRef` is `ComputedExpr` retains the structure of the underlying variables.

`ComputedExpr` enables the field / array access to determine the structure of the value with actual size of the targeted access. For example, when we are accessing `houses[1].rooms[2]`, it can check if the access is within the bounds of `houses` and `rooms`, by checking the structure and size of a targeted `ComputedExpr`.

In a sense, `ComputedExpr` can be seen as an resolved `Expr` with const variables replaced with actual values. 

### Types of ComputedExpr

```rust
pub enum ComputedExprKind<F, V>
where
    F: BackendField,
    V: BackendVar,
{
    /// Structures behind a custom struct can be recursive, so it embeds the ComputExpr.
    Struct(BTreeMap<String, ComputedExpr<F, V>>),
    /// Structures behind an array can be recursive, so it embeds the ComputExpr.
    Array(Vec<ComputedExpr<F, V>>),
    Bool(Var<F, V>),
    Field(Var<F, V>),
    /// Access to a variable in the scope.
    Access(Access<F, V>),
    /// Represents the results of a builtin function call.
    /// Because we don't know the exact type of the result, we store it as a Var.
    /// We may deprecate this once it is able to type check the builtin functions,
    /// so that the result can be inferred.
    FnCallResult(Var<F, V>),
}
```

### Assignment using ComputedExpr

Instead of using `VarOrRef` to narrow down and do the updating of `cvars`, we  use the `ComputedExpr` to represent the access path, which is iteratively determined when stepping through the computations of `Expr::FieldAccess` and `Expr::ArrayAccess`.

When it comes to the computation for `Expr::Assignment`, it can use the access path embeded in a `ComputedExprKind::Access` to locate a nested `ComputedExpr` to swap with a new one. Then it saves the `VarInfo` with same name `houses` as local variable but holding the updated tree of `ComputedExpr` that represents a new set of `cvars`.

### Use ComputedExpr in builtin functions

Psuedo code to show how the bit builtin functions could be implemented with the `ComputedExpr`:

```rust
fn to_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<ComputedExpr<B::Field, B::Var>>> {
    // should be two input vars
    assert_eq!(vars.len(), 2);

    // first var is the number of bits
    let num_var = &vars[0];
    assert!(matches!(num_var.expr.kind, ComputedExprKind::Field));

    // second var is the value to convert
    let val_var = &vars[1];
    assert!(matches!(val_var.expr.kind, ComputedExprKind::Field));

    // of only one field element for number of bits
    let var = &num_var.expr.clone().value();
    assert_eq!(var.len(), 1);
    let num_bits = &var[0];

    // of only one field element for value
    let var = &val_var.expr.clone().value();
    assert_eq!(var.len(), 1);
    let value = &var[0];

    // convert value to bits
    let mut bits = Vec::new();
    let mut lc = 0;
    let e2 = 1;
    for i in 0..num_bits {
        // bits[i] = (in >> i) & 1;
        let bit = compiler.backend.and(
            compiler.backend.shr(value, i),
            compiler.backend.constant(1),
        );
        let bit_ce = ComputedExpr::new_bool(bit);
        bits.push(bit_ce);

        // bits[i] * (bits[i] -1 ) === 0;
        let zero_mul = compiler.backend.mul(bit, compiler.backend.sub(bit, 1));
        compiler.backend.assert_eq_const(zero_mul, 0, span);

        // lc += bits[i] * e2;
        lc = compiler.backend.add(lc, compiler.backend.mul(bit, e2));

        e2 = e2 + e2;
    }

    compiler.backend.assert_var!(value, lc, span);

    // return the bits as an array type ComputedExpr.
    // this might improve checking the return type during circuit generation for the builtin functions, since this allows explicitly customizing the return type rather than always returning an array of cvars.
    Ok(Some(ComputedExpr::Array(bits)))
}
```

```rust
fn from_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<ComputedExpr<B::Field, B::Var>>> {
    // should be one input var
    assert_eq!(vars.len(), 1);

    let bits_var = &vars[0];

    // now we can make sure it is of array type explicity.
    // VarInfo::typ can also check the type.
    // But this can be useful for checking the actual array size of an array with const generic vars if needed. (by unwrapping the enum value)
    assert!(matches!(val_var.expr.kind, ComputedExprKind::Array));
    let ce = bits_var.expr.array_expr();

    let num_bits = ce.len();

    let mut lc = 0;
    let e2 = 1;
    // elements should be boolean type
    for i in 0..num_bits {
        let bit = &ce[i];
        assert!(matches!(bit.kind, ComputedExprKind::Bool));
        // lc += bits[i] * e2;
        lc = compiler.backend.add(lc, compiler.backend.mul(bit.value(), e2));
        e2 = e2 + e2;
    }

    Ok(Some(ComputedExpr::Field(lc)))
```
## Summary
This RFC is to support variable-length arguments and returned arrays in functions.

## Motivation
In the current implementation, array initialization requires explicitly stating all element values, which becomes impractical for large arrays. Furthermore, the lack of support for constant variables as array sizes limits the ability to write functions that generically handle arrays of varying sizes. This also necessitates hardcoding loop ranges, reducing flexibility.

Allowing to define array sizes with constant variables would enhance function reusability and improve modularization.

## Detailed design
One of the useful features of const generic is to allow the declaration of default array or return array with const variable as size. Before making it possible to unlock these features, we need a way to propagate the actual values for the constant variables. This is crucial for doing sanity checks during circuit generation, such as checking the bounds of array access.

To propagate the constant values, it needs to retain the structures for the variables from the bottom up. These retained structures would be also useful for builtin functions to do sanity checks on the inputs and outputs. For the support of const generic on types in the future, it would probably need to propagate the actual values in the same way.

### Glossary
- const: a keyword that can be used in different places. For example, when used in front of an argument name, in a function signature, it dictates that the function must be called with a value decided at compile time (for example, with a literal, or a global constant)"

### Resolving const variables
When the size of an array declaration is a literal number, the size can be resolved by calling the `size_of` on the array type. Then the `ExprKind::ArrayAccess` can check the bounds of the access by referencing the number returned from `size_of` call on an array type.

When it is a constant variable as array size, we can't rely on the `size_of` to determine the size of the array directly through the type anymore, as the type checker can't resolve these values under its current design. For example, the type checker can't determine wether the length of the array matches with the return type:

```rust
fn thing(const len: Field) -> [Field; len] {
  return [0; 3];
}
```

That is because the type checker doesn't know the actual value of `len`, which needs to be propagated when computing the `Expr` during circuit generation.

This RFC proposal is to defer the resolution of the actual values for the variable sizes of arrays to the circuit generation phase. The resolution can be done via propagating the actual sizes from the bottom up. This means to store the resolved values in a structured way, so they can be accessible in the computations of `compute_expr`, which is to compute the structured expressions. 

The constructions are done from the bottom up. For example `houses[1].rooms[2]`, the actual structure of `rooms` is determined by the `ExprKind::ArrayDeclaration` first, and then is embedded in the `houses` as a field.

Thus, instead of resolving the size through the `size_of` of a type, it needs to know the inner structure behind a `VarOrRef` to resolve the actual size. The problem is `VarOrRef` doesn't retain the inner structure when it is constructed from the bottom up. When it comes to the computation of the `Expr::ArrayAccess`, neither `VarOrRef` itself or the use of `size_of` of the type can provide the actual size of the array for checking the bounds.


### Introduce `ComputedExpr`

With a constant variable as array size, we need to propagate its value during circuit generation to check if the access is within the bounds of the array. 

We can replace the `VarOrRef` with a new struct that can keep track of the structure after it is constructed by `computed_expr`. It is called `ComputedExpr`, as it is a result of `compute_expr` function, which resolves and passes around the underlying values (`cvars`) via the `Expr`. The key difference from `VarOrRef` is that `ComputedExpr` retains the structure of the underlying variables, while they both represent a list of underlying `cvars`.

`ComputedExpr` enables the field / array access to determine the structure of the value with actual size of the targeted access. For example, when we are accessing `houses[1].rooms[2]`, it can check if the access is within the bounds of `houses` and `rooms`, by checking the structure and size of a targeted `ComputedExpr`.

In a sense, `ComputedExpr` can be seen as an resolved `Expr` with constant variables replaced with actual values. 

### Defintion of `ComputedExpr`
Here are the definitions of `ComputedExpr` and `ComputedExprKind`. The `ComputedExprKind` holds structural information built from underlying variables `cvars`.

```rust
pub struct ComputedExpr<F, V>
where
    F: BackendField,
    V: BackendVar,
{
    kind: ComputedExprKind<F, V>,
    span: Span,
}
```

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

### Assignment using `ComputedExpr`

Instead of using `VarOrRef` to narrow down and do the updating of `cvars`, we  use the `ComputedExpr` to represent the access path, which is iteratively determined when stepping through the computations of `Expr::FieldAccess` and `Expr::ArrayAccess`.

When it comes to the computation for `Expr::Assignment`, it can use the access path embedded in a `ComputedExprKind::Access` to locate a nested `ComputedExpr` to swap with a new one. Then it saves the `VarInfo` with same name `houses` as local variable but holding the updated tree of `ComputedExpr` that represents a new set of `cvars`.

### Examples
The following are examples of how const generic can be used in functions, once this RFC is implemented.

#### Declare a const argument
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

#### Builtin functions

Allow builtin functions to be defined with const arguments:

```rust
use std::to_bits;
use std::from_bits;

const num_bits = 8;

fn main() {
    let val1 = 101;
    let bits = to_bits(num_bits, val); // `bits` is an array of length `num_bits`
    let val2 = from_bits(bits); // `from_bits` builtin impl knows the length of the array
    assert_eq(val1, val2);
}
```

The `to_bits` builtin function accepts a const variable to determine the size of the return array.

```rust
fn to_bits(const num_bits: Field, val: Field) -> [Field; num_bits]
```

The `from_bits` builtin function can determine the number of bits to convert based on the information that the `ComputedExprKind::Array` embeds in the `bits` variable. 

#### Use ComputedExpr in builtin functions

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

    compiler.backend.assert_var(value, lc, span);

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
# Expressions

## Field accesses

A field access is an access to a field of a structure, by writing `struct.field`.
It is represented as an expression in noname:

```rust
pub enum ExprKind {
    // ...
    FieldAccess { lhs: Box<Expr>, rhs: Ident }
```

The reason why the left-hand side is also an expression, as opposed to just a variable pointing to a struct, is that we need to support code like this:

```rust
let a = b.c.d;
let e = f[3].g;
```

Note that there are other usecases that are not allowed at the moment for readability reasons.
For example we could have allowed the following to be valid noname:

```rust
let res = some_function().some_field;
```

but instead we require developers to write their logic in the following way:

```rust
let temp = some_function();
let res = temp.some_field;
```

In the example

```rust
let a = b.c.d;
```

the expression node representing the right hand side could be seen as:

```rust
ExprKind::FieldAccess {
    lhs: Expr { kind: ExprKind::FieldAccess { // x.y
        lhs: Expr { kind: ExprKind::Variable { name: "x" } },
        rhs: Ident { value: "y" },
    },
    rhs: Ident { value: "z" }, ///  [x.y].z
}
```

## Assignments

Imagine that we want to mutate a variable. 
For example:

```rust
x.y.z = 42;
x[4] = 25;
```

At some point the [circuit-writer]() would have to go through an expression node looking like this:

```rust
ExprKind::Assignment {
    lhs: /* the left hand side as an Expr */,
    rhs: Expr { kind: ExprKind::BigInt { value: 42 } },
}
```

At this point, the problem is that to go through each expression node, we use the following API, which only gets us a `Var`:

```rust
fn compute_expr(
    &mut self,
    global_env: &GlobalEnv,
    fn_env: &mut FnEnv,
    expr: &Expr,
) -> Result<Option<Var>> {
```

So parsing the `x.y` node would return a variable that either represents `x` or represents `y`.
The parent call would then use the result to produce `x.y.z` with a similar outcome.
Then, we would either have `x` or `z` (depending on the strategy we chose) when we reach the assignment expression node.
Not leaving us enough information to modify the variables of `x` in our [local function environment]().

What we really need when we reach the assignment node is the following:

- the name of the variable being modified (in both cases `x`)
- if the variable is mutable or not (it was defined with the `mut` keyword)
- the range of circuit variables in the `Var.cvars` of `x`, that the `x.y.z` field access, or the `x[42]` array access, represents.

## `VarOrRef` Overview

The VarOrRef enum is used to represent either a variable or a reference to a variable within expressions. 
Here is a concise overview:
```
pub enum VarOrRef<B: Backend> {
    /// A variable.
    Var(Var<B::Field, B::Var>),

    /// A reference to a variable, potentially narrowing down to a range within it.
    Ref {
        var_name: String,
        start: usize,
        len: usize,
    },
}
```

`Var`: Represents a complete variable in the environment.

`Ref`: Represents a reference to a variable, including:

- `var_name`: The name of the variable.
- `start`: The starting index of the slice or field.
- `len`: The length of the slice or field.

Every expression node in the AST is resolved as a `VarOrRef`, an enum that represents either a variable, or a reference to a variable.  The sole reason to use a reference is when the variable is **mutable**, in which case you must be able to go to the list of variables present in the scope and mutate the correct one (so that if some logic tries to mutate it, it can). That's why, a `var_name` is stored in a reference. We also pass a `(start, len)` tuple to handle **mutable slices**. As we need to remember exactly where we are in the original array. As a slice is a narrowing of an array, we must not lose track of which array we were looking at originally (as this is what needs to be mutated). This ensures accurate modification of the variable's state, maintaining the integrity of the mutable references.

### Circuit writer

To implement this in the circuit writer, we follow a common practice of tracking **references**:

```rust
/// Represents a variable in the circuit, or a reference to one.
/// Note that mutable variables are always passed as references,
/// as one needs to have access to the variable name to be able to reassign it in the environment.
pub enum VarOrRef {
    /// A [Var].
    Var(Var),

    /// A reference to a noname variable in the environment.
    /// Potentially narrowing it down to a range of cells in that variable.
    /// For example, `x[2]` would be represented with "x" and the range `(2, 1)`,
    /// if `x` is an array of `Field` elements.
    Ref {
        var_name: String,
        start: usize,
        len: usize,
    },
}
```

and we modify the [circuit-writer]() to always return a [`VarOrRef`]() when processing an expression node in the AST.

```admonish
While the type checker already checks if the `lhs` variable is mutable when it encounters an assignment expression,
the circuit-writer should do its best to pass references only when a variable is known to be mutable.
This way, if there is a bug in the type checker, this will turn unsafe code into a runtime error.
```

An array access, or a field access in a struct, is processed as a narrowing of the range we're referencing in the original variable:

```rust
impl VarOrRef {
    fn narrow(&self, start: usize, len: usize) -> Self {
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
                assert!(start + len < *old_len); // upper bound
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

### Type checker

While the type checker does not care about the range within a variable, it also needs to figure out if a variable is mutable or not.

That information is in two places:

1. it is stored under the variable's name in the local environment
2. it is also known when we look up a variable, and we can thus bubble it up to the parent expression nodes

Implementing solution 1. means bubbling up the variable name, in addition to the type, associated to an expression node.

Implementing solution 2. means bubbling up the mutability instead.

As it is possible that we might want to retrieve additional information in the future, we chose to implement solution 1. and carry the variable name in addition to type information when parsing the AST in the type checker.

# Type Checker

Noname uses a simple type checker to ensure that all types are consistent in the program.

For example, in code like:

```rust
let x = y + z;
```

the type checker will ensure that `y` and `z` are both field elements (because the operation `+` is used).

And in code like:

```rust
assert_eq(a, b);
```

the type checker will ensure that `a` and `b` are of the same types, since they are being compared.

## Type inference

The type checker can do some simple type inference. For example, in the following code:

```rust
let x = y + z;
```

the type of `x` is inferred to be the same as the type of `y` and `z`.

Inference is willingly kept naive, as more type inference would lead to less readable code.

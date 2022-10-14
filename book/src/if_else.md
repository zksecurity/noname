# If/else

The problem with if/else blocks like so:

```rust
if cond {
    // ...
} else {
    // ...
}
```

is that they are not supported by default in circuits (we're working on it!)

Instead, one can do simple ternary expressions as such:

```rust
let x = cond? var1 : var2;
```

where `var1` and `var2` can be variables, field accesses, and array accesses (basically anything that does not create a constraint and already exists).

We can represent this as the following expression node:

```rust
IfElse {
    cond: Expr,
    then_: Expr,
    else_: Expr,
}
```

This works because field accesses and array accesses are not abusively permissive in noname: you can't write a field access or array access that creates new constraints.
For example, the following code won't compile:

```rust
some_fn(arg).field
```

nor this

```rust
some_fn(arg)[0]
```

nor this:

```rust
([a[0] + a[1]])[0]
```

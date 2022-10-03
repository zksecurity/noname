# Literals and the `const` keyword

We want to be able to support a few things:

* writing numbers that will act as field elements directly in the code
* writing numbers that will act as relatively small numbers (for indexing into arrays for example) in the code
* writing functions that will accept constant arguments. For example to index into an array.

The first two points allow us to write things like that:

```rust
let x = 3;
assert_eq(y[5], 4);
```

The third point allows us to write things like that:

```rust
fn House.get(self room: Field) -> Field {
    return House.rooms[room];
}
```

Since `House` takes a `Field` here, it can in theory be used with anything during type checking.

This is bad. There are two solutions:

1. During type checking, when we realize that `room` is used to index into an array, we enforce that it must be a constant value during compilation.
2. We create a distinct type for constants and literals.

Approach 1. is not elegant, because it means that it is not clear from the signature of the function alone that the `room` argument must be a constant.
The user of the function will only get warned when trying to compile the program.

Approach 2. is interesting, because we already have such a type internally to track literals: a `BigInt`.
The name is a bit misleading in the case of array accesses, because we additionaly enforce that it is NOT a big integer, but rather a 32-bit integer (`u32` in Rust).

## Implementing a literal type

Approach 2 can be implemented in two ways:

a. Use a brand new type, like `BigInt` for literals.
b. Use a `const` attribute to indicate that it is a constant.

Approach a. is a bit clumsy in my opinion because the developer need to remember about a new type name, and understand the distinction with that and `Field`.

On the other hand, approach b. uses the `const` keyword which is already well-known in many compiled programming languages.

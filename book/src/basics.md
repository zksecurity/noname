# Basics

Noname is a language that closely resembles Rust.

For example, in the following program you can see a `main` function:

```rust
fn main(pub public_input: Field, private_input: Field) {
    let x = private_input + public_input;
    assert_eq(x, 2);
}
```

The only differences with Rust are:

* The `pub` keyword is used to mark _public_ inputs. By default all arguments are private.
* `assert_eq` is not a macro, there are no macros in noname.
* a `Field` type is used as main types everywhere. It is defined in `field.rs` to be the pasta Fp field (the base field of the Pallas curve). If these words mean nothing to you, just see `Field` as a large number. Ideally programs should be written without this type, but for now custom types do not exist.

## Use statements

Like in Rust, one can import other libraries via the `use` keyword:

```rust
use std::crypto;

fn main(pub public_input: Field, private_input: [Field; 2]) {
    let digest = crypto::poseidon(private_input);
    assert_eq(digest[0], public_input);
}
```

Note that currently, only built-in libraries (written in Rust) are working. 
In the future we'd like for other libraries to be written in noname.

## Boolean

TODO

## Custom types

TODO

## For loops

TODO

## If Else statements

TODO

## Functions

TODO

## Early returns

TODO

## Hints

TODO

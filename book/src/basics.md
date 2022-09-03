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

To run such a file, and assuming you have [Rust](https://rustup.rs/) installed, you can type in the terminal:

```
$ cargo run -- --path path/to/file.no --private-inputs '{"private_input": ["1"]}' --public-inputs '{"public_input": ["1"]}'
```

As you can see, inputs are passed with a JSON format, and the values are expected to be encoded in decimal numbers.

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

## Field

The `Field` type is the primitive type upon which all other types are built. 
It is good to know about it as it is used in many places, and is error prone: it does not match the size of commonly-found types like `u32` and `u64` and can have unexpected behaviors as it can overflow or underflow  without emitting an error.

Ideally, you should never use the `Field` type, but currently the library is quite limited and the ideal world is far away.

Note that you can define `Field` elements directly in the code by writing a decimal number directly. For example:

```rust
let x = 2;
assert(y, 4);
```

## Arrays

While there are no dynamic arrays (or vectors), you can use fixed-size arrays like in Rust.

For the moment, I believe that arrays can only be declared in a function argument as the following declaration hasn't been implemented yet:

```rust
let x = [1, 2, y];
```

## Boolean

Booleans are similar to Rust's boolean. They are currently the only built-in type besides `Field` and arrays.

```rust
let x = true;
let y = false;
assert(!(x & y));
```

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

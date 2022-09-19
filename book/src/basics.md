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

## Builtins and use statements

Some builtin functions are available by default:

* `assert_eq` to check that two field elements are equal
* `assert` to check that a condition is true.

Like in Rust, you can also import other libraries via the `use` keyword.
If you do this, you must know that you can only import a library, but not its functions (and types, and constants) directly.

For example, to use the poseidon function from the crypto library (or module), you must import `std::crypto` and then qualify your use of `crypto::poseidon`:

```rust
use std::crypto;

fn main(pub public_input: Field, private_input: [Field; 2]) {
    let digest = crypto::poseidon(private_input);
    assert_eq(digest[0], public_input);
}
```

Note that currently, only built-in libraries (written in Rust) are working. 
In the future we'd like for other libraries to be written in the noname language.

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

## Mutability

Variables are by default not mutable. To make a variable mutable, you must use the `mut` keyword:

```rust
let mut x = 1;
x = 2; // GOOD

let y = 1;
y = x + y; // BAD
```

## For loops

```rust
fn main(pub public_input: Field, private_input: [Field; 3]) {
    let mut sum = 0;

    for i in 0..3 {
        sum = sum + private_input[i];
    }

    assert_eq(sum, public_input);
}
```

## Constants

Like variables and function names, constants must be lowercase.

At the moment they can only represent field elements. Perhaps it would be nice to be able to represent different types in the future.

```rust
const player_one = 1;
const player_two = 2;

fn main(pub player: Field) -> Field {
    assert_eq(player_one, player);
    let next_player = player + 1;
    assert_eq(player_two, next_player);
    return next_player;
}
```

## If Else statements

TODO

## Functions

```rust
fn add(x: Field, y: Field) -> Field {
    return x + y;
}

fn double(x: Field) -> Field {
    return x + x;
}

fn main(pub one: Field) {
    let four = add(one, 3);
    assert_eq(four, 4);

    let eight = double(4);
    assert_eq(eight, double(four));
}
```

## Custom types

```rust
struct Thing {
    x: Field,
    y: Field,
}

fn main(pub x: Field, pub y: Field) {
    let thing = Thing {
        x: 1,
        y: 2,
    };
    
    assert_eq(thing.x, x);
    assert_eq(thing.y, y);
}
```

## Methods on custom types

```rust
struct Thing {
    x: Field,
    y: Field,
}

fn Thing.new(x: Field, y: Field) -> Thing {
    return Thing {
        x: x,
        y: y,
    };
}

fn Thing.verify(self, v: Field) {
    assert_eq(self.x, v);
    assert_eq(self.y, v + 1);
}

fn Thing.update_and_verify(self) {
    let new_thing = Thing {
        x: self.x + 1,
        y: self.y + 1,
    };

    new_thing.verify(2);
}

fn main(pub x: Field) {
    let thing = Thing.new(x, x + x);
    thing.update_and_verify();
}
```

technically you can even call static methods from a variable of that type:

```rust
let y = thing.new(3, 4);
```

```admonish
It's not necessarily pleasant to read, and we could prevent it by storing some meta information (`static_method: bool`) in the type checker, but it's not a big deal.
```

## Early returns

TODO

## Hints

TODO

## Shadowing

We forbid variable shadowing as much as we can.

For example, this should not work:

```rust
let x = 2;
let x = 3; // this won't compile

let y = 4;
for i in 0..4 {
    let y = i; // this won't compile either
}
```

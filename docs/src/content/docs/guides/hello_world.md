---
title: Hello World
description: Our first hello world example
---

Welcome to our introduction to **Noname Language**! This new language is designed with clarity and simplicity in mind, allowing you to clearly distinguish between public and private data in your functions.

Below is a simple example written in Noname Language:

```rust
fn main(pub public_input: Field, private_input: Field) {
    let xx = private_input + public_input;
    let yy = private_input * public_input;
    assert_eq(xx, yy);
}
```

What’s Happening Here?

**Function Definition.** The main function is defined with two parameters: a public input (via the `pub` keyword) and a private input (the default for any argument), both of type `Field`.

**Function Body.** The `assert_eq(xx, yy)` statement checks that the sum (`xx`) is equal to the product (`yy`). This simple assertion serves as a basic demonstration of the main way of writing zk circuits: **using assertions**. In zk circuits we never have to handle exceptions or errors, which is nice.

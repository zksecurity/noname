# Spans

To be able to efficiently track errors, we have a span type:

```rust
pub struct Span(pub usize, pub usize);
```

which represents a location in the original source code:

* the first number is the offset in the source code file
* the second number is the length of the span (e.g. 1 character)

We start tracking spans in the lexer, and then pass them around to the parser, and then to the compiler. Even gates and wirings have spans associated with them so that we can easily debug those.

## Filename

The filename is currently missing from the `Span`, it is annoying to add it as a `String` because then we can't easily copy the span around (`String` is not `Copy` but `Clone`).

One way to solve this, is to add the filenames in a `Hashmap<usize, String>`, and have the `usize` be in the `Span`.

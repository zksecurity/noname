# Spans

To be able to efficiently track errors, we have a span type:

```rust
pub type Span = (usize, usize);
```

which represents a location in the original source code:

* the first number is the offset in the source code file
* the second numebr if the length of the span (e.g. 1 character)

We start tracking spans in the lexer, and then pass them around to the parser, and then to the compiler. Even gates and wirings have spans associated with them so that we can easily debug those.

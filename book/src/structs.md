# Structs

User can define custom structs like so:

```rust
struct Thing {
    x: Field,
    y: Field,
}
```

and can declare and access such structs like so:

```rust
let thing = Thing { x: 1, y: 2 };
let z = thing.x + thing.y;
```

Internally, a struct is represented within the [`Var`](https://mimoo.github.io/noname/rustdoc/var/struct.Var.html) type.

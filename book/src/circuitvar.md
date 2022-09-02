# CircuitVar

We already have [`CellVar`](./cellvar.md)s, why have circuit vars? 
The distinction is a matter of level: 

* `CellVar`s are low-level: they track actual cells of the execution trace.
* `CircuitVar`s are a bit less low-level: they track variables that are made out of several field elements (and thus several `CellVar`s). For example, you can imagine that user could create types that take several field elements.

In the code, we define a variable as either a constant or a `CellVar`:

```rust
pub enum Var {
    /// Either a constant
    Constant(Constant),
    /// Or a [CircuitVar]
    CircuitVar(CellVars),
}
```

See the [Constant chapter](./constants.md) to see why constants are treated differently.

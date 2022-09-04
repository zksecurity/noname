# Vars

We already have [`CellVar`](./cellvar.md)s, why have vars? 
The distinction is a matter of level: 

* `CellVar`s are low-level: they track actual cells of the execution trace. When a single `CellVar` is assigned to multiple cells the wiring must make sure the cells are wired.
* `Var`s are a higher-level concept: they track variables that are created in the language either directly (`let x = ...`) or indirectly (in `x + (y + z)` the term `y + z` is stored under a `Var`)

While a `CellVar` represents a single field element, a `Var` can potentially represent several field elements (and as such several cells in the execution trace). 
Here are some examples of `Var`s:

```rust
// a constant
let x = 5;

// a field element that will be computed at runtime
let y = private_input + 1;

// a builtin type, like an array, or a bool
let z = [y, x, 6];

// or a custom type
// we don't support that yet
```

See the [Constant chapter](./constants.md) to see why constants are treated differently.

Internally, a `Var` is represented as such:

```rust
pub enum ConstOrCell {
    /// A constant value.
    Const(Constant),

    /// A cell in the execution trace.
    Cell(CellVar),
}

pub struct Var {
    pub value: Vec<ConstOrCell>,
    pub span: Span,
}
```

and the circuit-writer will always return a `Var` from an expression.

In our example of:

```rust
let x = t + (z + y);
```

the `z + y` is parsed as an expression (a binary operation involving `z` and `y`) and stored under a var `var1` (TODO: I believe it's called anonymous var? Because it doesn't have a name). 
Then `t + ...` is also parsed as another binary operation expression and stored under another var `var2`.
Finally the `let x = ...` is parsed as an assignment statement, and `x` is stored as a local variable associated to the right handside var `var2`.
See the [Scope chapter](./scope.md) for more information on local variables.

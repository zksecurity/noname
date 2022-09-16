# Vars

We already have [`CellVar`](./cellvar.md)s, why have [`Var`](https://mimoo.github.io/noname/rustdoc/var/struct.Var.html)s? 
The distinction is a matter of abstraction: 

* `CellVar`s are low-level: they track actual cells of the execution trace. When a single `CellVar` is assigned to multiple cells the wiring must make sure the cells are wired (so that they can only have the same value).
* `Var`s are a higher-level concept: they track variables that are created in the noname language either directly (e.g. `let x = 3`) or indirectly (e.g. in `x + (y + z)` the term `y + z` is stored under an anonymous `Var`)

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
let s = Thing { x, y };
```

Internally, a [`Var`](https://mimoo.github.io/noname/rustdoc/var/struct.Var.html) is represented as such:

```rust
/// A constant value created in a noname program
pub struct Constant {
    /// The actual value.
    pub value: Field,

    /// The span that created the constant.
    pub span: Span,
}

/// Represents a cell in the execution trace.
pub enum ConstOrCell {
    /// A constant value.
    Const(Constant),

    /// A cell in the execution trace.
    Cell(CellVar),
}

/// A variable in a program can have different shapes.
pub enum VarKind {
    /// We pack [Const] and [CellVar] in the same enum because we often branch on these.
    ConstOrCell(ConstOrCell),

    /// A struct is represented as a mapping between field names and other [VarKind]s.
    Struct(HashMap<String, VarKind>),

    /// An array or a tuple is represetend as a list of other [VarKind]s.
    ArrayOrTuple(Vec<VarKind>),
}

/// Represents a variable in the noname language, or an anonymous variable during computation of expressions.
pub struct Var {
    /// The type of variable.
    pub kind: VarKind,

    /// The span that created the variable.
    pub span: Span,
}
```

```admonish
Note: see the [Constant chapter](./constants.md) to see why constants are treated differently.
```

## Anonymous variable

Here's a short note on anonymous variable.

When circuit writer parses the [ast](./compilation.md), it will convert each expression into a `Var` (unless the expression does not compute to an actual value).

In our example:

```rust
let x = t + (z + y);
```

the `z + y` is parsed as an expression (a binary operation involving `z` and `y`) and stored under a var `var1`. 
Then `t + ...` is also parsed as another binary operation expression and stored under another var `var2`.
Finally the `let x = ...` is parsed as an assignment statement, and `x` is stored as a local variable associated to the right handside var `var2`.

```admonish
See the [Scope chapter](./scope.md) for more information on local variables.
```

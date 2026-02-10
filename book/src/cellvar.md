# CellVar

A `CellVar` type is a type that represents an internal variable. Importantly, it is named after the fact that it relates to a specific cell, or even multiple cells if they will have the same value (using some wiring), in the execution trace.

A `CellVar` looks like this:

```rust
pub struct CellVar {
    index: usize,
    span: Span,
}
```

It is tracked using a `usize`, which is just a counter that the compiler increments every time a new `CellVar` is created.

A `CellVar` is created via the `new_internal_var` function which does two things: increments the variable counter, and stores some information on how to compute it (which will be useful during witness generation)

```rust
pub fn new_internal_var(&mut self, val: Value, span: Span) -> CellVar {
    // create new var
    let var = CellVar::new(self.next_variable, span);
    self.next_variable += 1;

    // store it in the compiler
    self.vars_to_value.insert(var, val);

    var
}
```

a `Value` tells us how to compute the `CellVar` during witness generation:

```rust
pub enum Value {
    /// Either it's a hint and can be computed from the outside.
    Hint(Box<dyn Fn(&Compiler, &mut WitnessEnv) -> Result<Field>>),

    /// Or it's a constant (for example, I wrote `2` in the code).
    Constant(Field),

    /// Or it's a linear combination of internal circuit variables (+ a constant).
    LinearCombination(Vec<(Field, CellVar)>, Field),

    /// A public or private input to the function
    /// There's an index associated to a variable name, as the variable could be composed of several field elements.
    External(String, usize),

    /// A public output.
    /// This is tracked separately as public outputs as it needs to be computed later.
    PublicOutput(Option<CellVar>),
}
```

Note: a `CellVar` is potentially not directly added to the rows of the execution trace. 
For example, a private input is converted directly to a (number of) `CellVar`(s), 
but only added to the rows when it appears in a constraint for the first time.

As the final step of the compilation, we double-check that all `CellVar`s have appeared in the rows of the execution trace at some point. If they haven't, it can mean two things:

* A private or public input was never used in the circuit. In this case we return an error to the user.
* There is a bug in the compiler. In this case we panic.

TODO: explain the LinearCombination. I think we only need an `Add((Field, Var), (Field, Var), Field)`

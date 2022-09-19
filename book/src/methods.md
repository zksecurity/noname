# Methods

## A method on what?

There's one problem when handling methods in the circuit-writer: how do you know where the code of that method is? For example:

```rust
let thing = Thing { x: 1, y: 2 };
let z = thing.valid(3);
```

at this point the circuit-writer knows that `Thing` has a method called `valid`, but will still wonder what the type of `thing` is.

due to this, the circuit-writer needs to store the type of local variables in scope. And this is why `FnEnv` also keeps track of the type of local variables:

```rust
/// Is used to store functions' scoped variables.
/// This include inputs and output of the function,  as well as local variables.
/// You can think of it as a function call stack.
pub struct FnEnv {
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Used by the private and public inputs,
    /// and any other external variables created in the circuit
    /// This needs to be garbage collected when we exit a scope.
    /// Note: The `usize` is the scope in which the variable was created.
    vars: HashMap<String, (usize, VarInfo)>,
}

/// Information about a variable.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// The variable.
    pub var: Var,

    /// We keep track of the type of variables, eventhough we're not in the typechecker anymore,
    /// because we need to know the type for method calls.
    pub typ: TyKind,
}
```

This still doesn't fix our problem. In the line:

```rust
let thing = Thing { x: 1, y: 2 };
```

the local variable `thing` is stored, but the right hand side is computed via the `compute_expr()` function which will go through the AST and potentially create different anonymous variables until it can compute a value.

There's three ways to solve this:

1. Either the type checker stores type information about each expression it parses. This is what the Rust compiler does I believe: each `Expr` AST node has a unique node identifier that can be used to search type information in a map.
2. Or, more simply, the circuit-writer's `compute_expr()` function that returns an `Option<Var>` could be modified to return `Option<VarInfo>`. This is a bit annoying as we're recomputing things we've done in the type checker.
3. A variant of the previous option is to change `Var` so that it also contain a type (might as well).

So we implement option 1: the type checker now stores the type information of each `Expr` node in the AST under a hashmap that is later passed to the circuit-writer:

```rust
/// The environment we use to type check a noname program.
pub struct TypedGlobalEnv {
    // ...

    /// Mapping from node id to TyKind.
    /// This can be used by the circuit-writer when it needs type information.
    node_types: HashMap<usize, TyKind>,
}
```

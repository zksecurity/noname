# Witness Generation

Witness generation is the process of creating the execution trace table during proving. 
The execution trace table is then passed to the [kimchi]() proof system which will create the final proof.

The code creates a series of instructions during compilation for the witness generation to follow.
These instructions are stored as two different fields:

```rust
pub struct Compiler {
    // ...

    /// This is how you compute the value of each variable, for witness generation.
    pub witness_vars: HashMap<CellVar, Value>,

    // ...

    /// This is used to compute the witness row by row.
    pub rows_of_vars: Vec<Vec<Option<CellVar>>>,

    // ...
}
```

`rows_of_vars` can essentially be seen as the execution trace table, containing variables instead of values.

The witness generation goes as follows:

1. Each rows in `rows_of_vars` is looked at one by one 
2. For each `CellVar` in the row:
   1. If it is set, it is evaluated using the `Value` stored in `witness_vars`.
   2. If it set to `None`, it is simply evaluated as `0`.
3. Once the row is created, it is checked for correctness by checking what gate was used in the row. Note that this is only true for the generic gate, as we trust built-in gadgets to produce correct values. For example, `assert(x, 2)` will be checked because it is using the generic gate, but `let y = poseidon(x)` won't be because we trust the poseidon gate to be correct (and if there is a bug there, kimchi will still catch it).

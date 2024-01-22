# Public Outputs

Public outputs are usually part of the public inputs in Plonk.

In noname, public outputs are treated differently than the public inputs for one reason: unlike (real) public inputs they cannot be computed directly during witness generation (proving).

This is because public inputs are listed first in the circuit. During witness generation, we go through each rows and evaluate the values of the cells to construct the execution trace. 
When we reach the public output part of the public input, we do not yet have enough information to construct the values.
Thus, we ignore them, and fill them later on.

During the compilation, we create `CellVars` to keep track of the public output:

```rust
pub struct Compiler {
    // ...

    /// If a public output is set, this will be used to store its [CircuitVar] (cvar).
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub public_output: Option<CellVars>,
}
```

During witness generation (see the [Witness Generation chapter](./witness-generation.md)), we indeed defer computation the first time we go through the public output rows:

```rust
let val = if let Some(var) = var {
    // if it's a public output, defer it's computation
    if matches!(self.witness_vars[&var], Value::PublicOutput(_)) {
        public_outputs_vars.push((row, *var));
        Field::zero()
    } else {
        self.compute_var(&mut env, *var)?
    }
} else {
    Field::zero()
};
witness_row[col] = val;
```

and at the end we go back to them:

```rust
// compute public output at last
let mut public_output = vec![];

for (row, var) in public_outputs_vars {
    let val = self.compute_var(&mut env, var)?;
    witness[row][0] = val;
    public_output.push(val);
}
```

and finally we return the public output to the prover so that they can send it to the verifier, as well as the "full public input" which is the concatenation of the public input and the public output (needed to finalized the proof):

```rust
// return the public output separately as well
Ok((Witness(witness), full_public_inputs, public_output))
```

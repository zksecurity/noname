# Constants

Developers write constants in their code all the time. For example, the following code has two constants `2` and `4`:

```rust
let x = 2 + y;
assert_eq(x, 4);
```

It is important that constants are tracked differently than `CellVar`s for several reasons:

* It is sometimes useless to constrain them directly. For example, in `let x = 3 + 7;` you can see that we should not constrain `3` and `7` separately, but rather the result `10`.
* They can be cached to avoid creating several constraints for the same constant.

Currently a constant appears in the circuit via a simple generic gate, and is not cached:

```rust
pub fn add_constant(&mut self, value: Field, span: Span) -> CellVar {
    let var = self.new_internal_var(Value::Constant(value), span);

    let zero = Field::zero();
    self.add_gate(
        GateKind::DoubleGeneric,
        vec![Some(var)],
        vec![Field::one(), zero, zero, zero, value.neg()],
        span,
    );

    var
}
```

Note that the `Value` keep track of the constant as well.

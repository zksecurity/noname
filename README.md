# I'm just toying around here, no idea what I'm doing.

**No really, this is all for me, not for you.**

The idea: a rust-inspired programming language to write circuits for [kimchi](https://github.com/o1-labs/proof-systems).

**Status: see [#roadmap]**

work for very simple examples. See [/data](/data)

For example, here's a circuit that has one public input and one private input, checks that they can add up to 2, then return their sum with 6 as public output:

```rust
fn main(pub public_input: Field, private_input: Field) -> Field {
    let x = private_input + public_input;
    assert_eq(x, 2);
    let y = x + 6;
    return y;
}
```

You can compile it with the following command:

```console
$ cargo run -- --path data/arithmetic.no --debug
```

Which will print the assembly, as well as try to create and verify a proof to make sure everything works. The assembly should look like this:

```
@ noname.0.1.0

DoubleGeneric<1>
DoubleGeneric<1,1,-1>
DoubleGeneric<1,0,0,0,-2>
DoubleGeneric<1,-1>
(0,0) -> (1,1)
(1,2) -> (3,1)
(2,0) -> (3,0)
```

If you run the command with `--debug` it should show you what lines created what gates:

```
--------------------------------------------------------------------------------
1: fn main(pub public_input: Field, private_input: Field) {
               ^^^^^^^^^^^^
DoubleGeneric<1>
--------------------------------------------------------------------------------
2:     let x = private_input + public_input;
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DoubleGeneric<1,1,-1>
--------------------------------------------------------------------------------
3:     assert_eq(x, 2);
                    ^
DoubleGeneric<1,0,0,0,-2>
--------------------------------------------------------------------------------
3:     assert_eq(x, 2);
       ^^^^^^^^^^^^^^^
DoubleGeneric<1,-1>
```

## Roadmap

Roadmap of the proof of concept:

- [x] [`arithmetic.no`](/data/arithmetic.no) (simple arithmetic)
- [x] `[public_output.no`](/data/public_output.no) (returns a public output)
- [x] `[poseidon.no`](/data/poseidon.no) (uses the poseidon function)
- [ ] bool.no
- [ ] `[types.no`](/data/types.no) (uses custom types)
- [ ] sudoku.no

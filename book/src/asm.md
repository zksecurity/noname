# noname ASM

The circuit that noname compiles to can be serialized as a simple ASM language. For example the following noname program:

```rust
fn main(pub public_input: Field, private_input: Field) {
    let x = private_input + public_input;
    assert_eq(x, 2);
}
```

will be compiled to the following noname asm:

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

which includes:

* **the version** of noname used to compile this circuit. This is important as the prover needs to know what version of noname to use to prove executions of this circuit.
* **a list of gates** and how they are tweaked (the values in the brackets).
* **a list of wires** which is canonically ordered so that every compilation gives the same resulting noname asm.

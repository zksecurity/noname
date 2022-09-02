# Noname

This is **work-in-progress** to implement a high-level (rust-inspired) programming language to write zkapps for [kimchi](https://github.com/o1-labs/proof-systems).

If you don't know what zero-knowledge proofs, zkapps, or kimchi are, check out [that blogpost](https://minaprotocol.com/blog/kimchi-the-latest-update-to-minas-proof-system).

## Examples

For example, here's a circuit that has one public input and one private input, checks that they can add up to 2, then return their sum with 6 as public output:

```rust
use std::crypto;

fn main(pub public_input: Field, private_input: [Field; 2]) {
    let x = private_input[0] + private_input[1];
    assert_eq(x, 2);
    
    let digest = crypto::poseidon(private_input);
    assert_eq(digest[0], public_input);
}
```

You can compile it with the following command:

```console
$ cargo run -- --path data/example.no --private-inputs '{"private_input": ["1", "1"]}' --public-inputs '{"public_input": ["3654913405619483358804575553468071097765421484960111776885779739261304758583"]}' --debug
```

Which will print the assembly, as well as try to create and verify a proof to make sure everything works. The assembly should look like this:

```
@ noname.0.1.0

c0 = -7792942617772573725741879823703654500237496169155240735183726605099215774906
// (truncated)
c164 = 10888828634279127981352133512429657747610298502219125571406085952954136470354
DoubleGeneric<1>
DoubleGeneric<1,0,0,0,0>
Poseidon<c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14>
Poseidon<c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29>
Poseidon<c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44>
Poseidon<c45,c46,c47,c48,c49,c50,c51,c52,c53,c54,c55,c56,c57,c58,c59>
Poseidon<c60,c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74>
Poseidon<c75,c76,c77,c78,c79,c80,c81,c82,c83,c84,c85,c86,c87,c88,c89>
Poseidon<c90,c91,c92,c93,c94,c95,c96,c97,c98,c99,c100,c101,c102,c103,c104>
Poseidon<c105,c106,c107,c108,c109,c110,c111,c112,c113,c114,c115,c116,c117,c118,c119>
Poseidon<c120,c121,c122,c123,c124,c125,c126,c127,c128,c129,c130,c131,c132,c133,c134>
Poseidon<c135,c136,c137,c138,c139,c140,c141,c142,c143,c144,c145,c146,c147,c148,c149>
Poseidon<c150,c151,c152,c153,c154,c155,c156,c157,c158,c159,c160,c161,c162,c163,c164>
DoubleGeneric<>
DoubleGeneric<1,-1>
(0,0) -> (14,1)
(1,0) -> (2,2)
(13,0) -> (14,0)
```

If you run the command with `--debug` it should show you what lines created what gates:

```
--------------------------------------------------------------------------------
3: fn main(pub public_input: Field, private_input: [Field; 2]) {
               ^^^^^^^^^^^^
DoubleGeneric<1>
--------------------------------------------------------------------------------
4:     let x = private_input[0] + private_input[1];
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DoubleGeneric<1,1,-1>
--------------------------------------------------------------------------------
5:     assert_eq(x, 2);
                    ^
DoubleGeneric<1,0,0,0,-2>
--------------------------------------------------------------------------------
5:     assert_eq(x, 2);
       ^^^^^^^^^^^^^^^
DoubleGeneric<1,-1>
```

and what lines created what wiring:

```
7:     let digest = crypto::poseidon(private_input);
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(4,0) -> (5,2)
--------------------------------------------------------------------------------
8:     assert_eq(digest[0], public_input);
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(16,0) -> (17,0)
```

If you pass an invalid input it should fail with an error:

```
$ cargo run -- --path data/example.no --private-inputs '{"private_input": ["2", "1"]}' --public-inputs '{"public_input": ["3654913405619483358804575553468071097765421484960111776885779739261304758583"]}'26177265001502838070204204
```

<img width="487" alt="Screen Shot 2022-09-02 at 12 08 41 PM" src="https://user-images.githubusercontent.com/1316043/188221355-4342b99c-3894-45f9-8bad-0f9477d93a63.png">


## Roadmap

Roadmap of the proof of concept:

- [x] [`arithmetic.no`](/data/arithmetic.no) (simple arithmetic)
- [x] `[public_output.no`](/data/public_output.no) (returns a public output)
- [x] `[poseidon.no`](/data/poseidon.no) (uses the poseidon function)
- [ ] make sure wiring debug shows you the var location, not the expression
- [ ] bool.no
- [ ] `[types.no`](/data/types.no) (uses custom types)
- [ ] sudoku.no

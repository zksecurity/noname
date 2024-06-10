# Noname

This is **work-in-progress** to implement a high-level (Rust and Golang-inspired) programming language to write zkapps for [kimchi](https://github.com/o1-labs/proof-systems) or [SnarkJS](https://github.com/iden3/snarkjs). The supported arithmetic backends are:
 - Kimchi 
 - R1CS

If you don't know what zero-knowledge proofs, zkapps, or kimchi are, check out [that blogpost](https://minaprotocol.com/blog/kimchi-the-latest-update-to-minas-proof-system).

You can read more about the project on the noname book: https://zksecurity.github.io/noname or my [series of blogposts](https://cryptologie.net/article/573).

I invite you to try to [install](#installation) and [play](#usage) with noname. Please provide feedback and suggestions via the [issues](https://github.com/zksecurity/noname/issues).

## Examples

For example, here's a circuit that has one public input and one private input:

```rust
use std::crypto;

fn main(pub public_input: Field, private_input: [Field; 2]) {
    // checks that they add up to 2
    let x = private_input[0] + private_input[1];
    assert_eq(x, 2);
    
    // checks that one is the hash of the other
    let digest = crypto::poseidon(private_input);
    assert_eq(digest[0], public_input);
}
```

You can compile it with the following command:

```console
$ noname test --path examples/example.no --private-inputs '{"private_input": ["1", "1"]}' --public-inputs '{"public_input": "3654913405619483358804575553468071097765421484960111776885779739261304758583"}' --debug
```

Which will print the assembly, as well as try to create and verify a proof to make sure everything works. The assembly should look like this:

```
@ noname.0.7.0

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

<img width="871" alt="Screen Shot 2022-11-11 at 11 01 45 PM" src="https://user-images.githubusercontent.com/1316043/201461923-8d6d6756-9faa-40fe-8f71-16334a4cb98d.png">

and what lines created what wiring:

<img width="871" alt="Screen Shot 2022-11-11 at 11 02 38 PM" src="https://user-images.githubusercontent.com/1316043/201461945-09121d99-1e7c-4204-962f-99cb24e26b50.png">

If you pass an invalid input it should fail with an error:

```
$ noname test --path examples/example.no --private-inputs '{"private_input": ["2", "1"]}' --public-inputs '{"public_input": "3654913405619483358804575553468071097765421484960111776885779739261304758583"}'
```

<img width="864" alt="Screen Shot 2022-11-11 at 11 03 09 PM" src="https://user-images.githubusercontent.com/1316043/201461958-43677bef-d251-4075-8314-3b22fc3ba05c.png">

## Installation

As this is work in progress, you need to install the compiler from source using [cargo](https://rustup.rs/). 
You can do this by running the following command:

```console
$ cargo install --git https://www.github.com/zksecurity/noname
```

## Usage

Simply write `noname` in the console to get access to the CLI:

```console
$ noname
```

To create a project, use `noname new` to create a project in a new directory, or `noname init` to initialize an existing directory. For example:

```
$ noname new --path my_project 
```

This will create a `Noname.toml` manifest file, which contains the name of your project (which must follow a Github `user/repo` format) as well as dependencies you're using (following the same format, as they are retrieved from Github).

This will also create a `src` directory, which contains a `main.no` file, which is the entry point of your program. If you want to create a library, pass the `--lib` flag to the `new` or `init` command of `noname`, and it will create a `lib.no` file instead.

```
$ tree
.
├── Noname.toml
└── src
    └── main.no
```

You can then use the following command to check the correctness of your code (and its dependencies):

```
$ noname check
```

or you can test a full run with:

```
$ noname test
```

which will attempt to create a proof and verify it. See the [examples](#examples) section to see how to use it.

## Current limitations

Currently there are no commands to compile a program and produce the compiled prover and verifier parameters. There is also no command to produce a serializable proof.

It's not hard, it's just hasn't been prioritized.

You can also not use third-party libraries (eventhough the CLI might suggest that it works). I think it's close to be working though.

In general, there's a large number of missing features. I will prioritize what I think needs to be done, but if you have a feature request, please open an [issue](https://github.com/zksecurity/noname/issues).

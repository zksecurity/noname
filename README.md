![Screenshot 2024-07-17 at 2 14 40 PM](https://github.com/user-attachments/assets/b2b84f62-3ac6-45c9-a005-b5b764916b6b)

Noname is a high-level programming language inspired by Rust and Golang to write zero-knowledge applications. The language can support multiple constraint systems, and currently compiles down to R1CS (for [SnarkJS](https://github.com/iden3/snarkjs)) and Plonk (for [kimchi](https://github.com/o1-labs/proof-systems)).

```rust
fn main(pub public_input: Field, private_input: Field) -> Bool {
    let xx = private_input + public_input;
    assert_eq(xx, 2);
    let yy = xx + 6;
    return yy == 8;
}
```

> [!IMPORTANT]
> Noname is currently in Beta, and there are a large number of known limitations. Please check the [issues](https://github.com/zksecurity/noname/issues) if you find something that doesn't work, or if you want to start contributing to this project!

You can run the above example with the following command:

```console
$ noname test --path examples/public_output_bool.no --private-inputs '{"private_input": "1"}' --public-inputs '{"public_input": "1"}' --debug
```

On particularity of noname is the `--debug` option that shows you how the code relates to the compiled constraints:

<img width="871" alt="Screen Shot 2022-11-11 at 11 01 45 PM" src="https://user-images.githubusercontent.com/1316043/201461923-8d6d6756-9faa-40fe-8f71-16334a4cb98d.png">

## Quick Start

To get started with no strings attached, you can use the [Noname Code Playground](https://noname-playground.xyz) to write and run your code in the browser. Alternatively, you can install the compiler on your system.

You need to install the compiler from source using [cargo](https://rustup.rs/).  You can do this by running the following command:

```console
$ cargo install --git https://www.github.com/zksecurity/noname
```

Then simply write `noname` in the console to get access to the CLI. See the [Usage](#usage) section for more information on usage.

```console
$ noname
```

[Learn more about how to use noname in the deepwiki documentation](https://deepwiki.com/zksecurity/noname/1-overview).

## More Resources

We have a lot of resources to learn and understand how noname works:

- [Noname Book](https://zksecurity.github.io/noname): A high-level language to write circuits using a zero-knowledge proof system
- [Noname Playground](https://noname-playground.xyz): Write, run, and prove Noname code in the browser
- [Kimchi](https://minaprotocol.com/blog/kimchi-the-latest-update-to-minas-proof-system): The latest update to Mina’s proof system
- Series of [blog posts](https://cryptologie.net/article/573) to read more about the noname
- [Noname meets Ethereum](https://www.zksecurity.xyz/blog/posts/noname-r1cs/): Integration with SnarkJS
- [Noname Code Walkthrough](https://www.youtube.com/live/pQer-ua73Vo)
- Join us on the [Noname Telegram](https://t.me/+VSChAOmJQgQzODcx)

## Usage

Once [noname is installed on your system](#installation), use `noname new` to create a project in a new directory, or `noname init` to initialize an existing directory. For example:

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

which will attempt to create a proof and verify it. See the [examples](https://github.com/zksecurity/noname/tree/main/examples) folder to see how to use it.


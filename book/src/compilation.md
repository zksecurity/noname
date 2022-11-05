# Compilation

The compilation of noname programs goes through the following flow:

1. **Lexer**. A lexer (`lexer.rs`) is used to parse the source code into a list of tokens. This is pretty primitive, but will detect some minor syntax issues.
2. **Parser**. A parser (`parser.rs`) is used to parse meaning from the code. It will convert the tokens output by the lexer into an [abstract syntax tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) using strong types like [`Statement`]() and [`Expression`]() (TODO: link to rust doc). It will also error if some code does not make sense according to the grammar (see the [Grammar chapter](grammar.md)).
3. **Type checking**. A type checker (`type_checker.rs`) takes the AST produced by the parser and does import resolution and type checking: 
   - **Built-in functions**. Functions like `assert_eq` are injected into the environment.
   - **Custom imports**. Modules imported via the `use` keyword are resolved and added to the environment. For now, these can only be built-in functions, and noname functions or libraries are not supported (of course it is essential to support them in the future).
   - **Type checking**. The type checker verifies that the types of each variables and expressions in the AST make sense. It is a very simple type checker that can do some simple type inference. Temporary type information (type of an expression) is not stored, and is thrown away as soon as the type checker can afford it. a TAST for typed AST is returned, but it mostly contains resolved imports and most type information has been thrown away.
4. **Gate construction**. The TAST produced by the type checker is passed to the circuit writer (`circuit_writer.rs`), also called the constraint writer, which goes through it one more time and converts it into:
   - **compiled circuit**: a series of gates and wires
   - **prover instructions**: instructions on how to run the function for the witness generation (used by the prover)

A simple ASM language is also used, and the circuit can be encoded in this language. See the [ASM chapter](asm.md).

## Terminology

A note on topology:

* **functions**: noname functions each contain their scope and can be interacted with their interface (arguments and return value)
* **module/program**: a noname module is a single file (this is a nice current limitation of noname) containing functions, constants, and structures. 
* **library**: a noname library is a module/program without a `main()` function, as well as dependencies (other libraries)
* **executable**: a noname executable is like a library, except that its module/program has a `main()` function.

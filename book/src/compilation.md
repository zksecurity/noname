# Compilation

The compilation of noname programs goes through the following flow:

1. **Lexer**. A lexer (`lexer.rs`) is used to parse the source code into a list of tokens. This is pretty primitive, but will detect some minor syntax issues.
2. **Parser**. A parser (`parser.rs`) is used to parse meaning from the code. It will convert the tokens output by the lexer into an [abstract syntax tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) using strong types like [`Statement`]() and [`Expression`]() (TODO: link to rust doc). It will also error if some code does not make sense according to the grammar (see the [Grammar chapter](grammar.md)).
3. **AST**. The compilation of the AST into a circuit happens for the most part in `ast.rs`.

Next we describe the AST phase of the compilation:

1. **Built-in functions**. Functions like `assert_eq` are injected into the environment.
2. **Custom imports**. Modules imported via the `use` keyword are resolved and added to the environment. For now, these can only be built-in functions, and noname functions or libraries are not supported (of course it is essential to support them in the future). Note that at this moment this is done in the type checker (next section) but should be moved out.
3. **Type checking**. The type checker (`type_checker.rs`) verifies that the types of each variables and expressions in the AST make sense. It is a very simple type checker that can do some simple type inference. Temporary type information (type of an expression) is not stored, and is thrown away as soon as the type checker can afford it.
4. **Gate construction**. The AST is parsed one more time and converted into:
   - **compiled circuit**: a series of gates and wires
   - **prover instructions**: instructions on how to run the function for the witness generation (used by the prover)

A simple ASM language is also used, and the circuit can be encoded in this language. See the [ASM chapter](asm.md).

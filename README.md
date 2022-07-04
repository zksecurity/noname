# I'm just toying around here, no idea what I'm doing.

No really, this is all for me, not for you.

The idea: a rust-inspired programming language to write circuits for kimchi.

Status: I can parse an extremely simple circuit, produce the circuit (in some made-up asm language), link each gate to the source, and generate a witness for that circuit.

![image](https://user-images.githubusercontent.com/1316043/175832784-b77ae752-4513-4bae-9268-0d75eb558495.png)

## Concept

kimchi is reused for the following flows:

- circuit -> kimchi compiler -> prover/verifier index
- prover index + witness -> kimchi prover -> proof
- verifier index + proof -> kimchi verifier -> true/false

Some mentras:

- I like Rust, let's make something that looks like Rust
- I like Golang also, let's take things from there as well:
  - import a module not its functions and 1-depth qualify everything
  - if there's two ways to write something, then that's bad (for example, type inference or writing the type manually? Don't give the option of the latter (eventhough go does, go often has one way to do something))
- enforcing formatting (and not making it customizable) would be great :D
- easy to compile/use something on any machine (like go/cargo)
- make it composable (let people create their libraries)

## Roadmap

Roadmap of the proof of concept:

- [x] code -> lexer -> token
- [x] token -> parser -> AST
- [ ] AST -> semantic analysis -> AST + type info
  - [ ] type checking
  - [ ] name binding
  - [ ] flow checking (CFG?)
- [ ] AST + type info -> compile -> circuit + witness info
- [ ] witness info -> interpreter -> witness

Files I should be able to parse:

- [x] `arithmetic.no`
- [ ] `public_output.no`
- [ ] `poseidon.no`
- [ ] `types.no`

More specific tasks:

- [x] fix the bug (the 2 isn't constrained in arithmetic.no). Probably I need to handle constants differently (I don't constrain them yet).
- [ ] the witness should be verified as it is created (either we run the circuit with the witness, or when we construct the circuit we also info on what needs to be checked when the witness is created? the latter solution seems more elegant/efficient)
- [ ] handle function call in a statement differently? I could simply say that a statement can be an expression, only if it's a function call (to dedup code, although semantically I don't like it... maybe better to just factor out the code in a function)
- [ ] handle public output when generating witness

## Questions

- interestingly, it seems like other languages have expression as statements, but check that the expression actually has a type (different than unit). Why do this just for a function call expression type? I find it better to include the function call as part of a statement as well as part of an expression, voila
- interestingly, looks like the type system either creates a different AST or stores the information elsewhere. I think for our PoC we can either: 1) not store it or 2) store it within the Expr as an `Option<TyKind>`
- do I need an ASG? CFG?

## Relevant resources

- maybe I should have used one of the easy parser library: https://github.com/lalrpop/lalrpop and https://github.com/pest-parser/pest (but a bit too late for that)
- rustc has some explanation on its inners: https://rustc-dev-guide.rust-lang.org/the-parser.html
- youtube course on compilers from Nicolas Laurent https://www.youtube.com/watch?v=hvGPtdNSvt8&list=PLOech0kWpH8-njQpmSNGSiQBPUvl8v3IM&index=2
- Leo whitepaper https://eprint.iacr.org/2021/651
- Move paper?


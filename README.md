# I'm just toying around here, no idea what I'm doing.

No really, this is all for me, not for you.

It'd be good to write a proof of concept that works, even if ugly, to compile one very simple circuit. For now, I'm trying this with `data/arithmetic.no`:

```rust
fn main(pub public_input: Field, private_input: Field) {
    let x = private_input + private_input;
    assert_eq(x, 2);
}
```

Roadmap of the proof of concept:

- [x] code -> lexer -> token
- [x] token -> parser -> AST
- [ ] AST -> semantic analysis -> AST + type info
  - [ ] type checking
  - [ ] name binding
  - [ ] flow checking (CFG?)
- [ ] AST + type info -> compile -> circuit + witness info
- [ ] witness info -> interpreter -> witness

then I reuse kimchi for the following flows:

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

Questions:

- interestingly, it seems like other languages have expression as statements, but check that the expression actually has a type (different than unit). Why do this just for a function call expression type? I find it better to include the function call as part of a statement as well as part of an expression, voila
- interestingly, looks like the type system either creates a different AST or stores the information elsewhere. I think for our PoC we can either: 1) not store it or 2) store it within the Expr as an `Option<TyKind>`
- do I need an ASG? CFG?

More resources that I found useful or relevant:

- maybe I should have used one of the easy parser library: https://github.com/lalrpop/lalrpop and https://github.com/pest-parser/pest (but a bit too late for that)
- rustc has some explanation on its inners: https://rustc-dev-guide.rust-lang.org/the-parser.html
- youtube course on compilers from Nicolas Laurent https://www.youtube.com/watch?v=hvGPtdNSvt8&list=PLOech0kWpH8-njQpmSNGSiQBPUvl8v3IM&index=2
- Leo whitepaper https://eprint.iacr.org/2021/651
- Move paper?


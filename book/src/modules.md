# Modules

In noname, the concept of a module is basically a file. A project either is a binary (`main.no`) or a library (`lib.no`). That's it.

A binary or a library can use other libraries by importing them. To do that, a binary or library's manifest file `Noname.toml` must contain a `dependencies` key listing all the other libraries as Github handles like `user/repo` (e.g. `mimoo/sudoku`).
Libraries will then be retrieved from Github. 

```admonish
Currently there is no versioning. Not because it's not important, but because I haven't had the time to implement it.
```

Each library can be imported in code with the following command:

```
use module::lib;
```

For example, currently you automatically have access to the `std` module:

```
use std::crypto;

fn main(pub digest: [Field; 2]) {
    let expected_digest = crypto::poseidon([1, 2]);
    assert_eq(expected_digest, digest);
}
```

Each library is seen as a module, and different modules might have the same name:

```
use a::some_lib;
use b::some_lib;
```

There is currently no solution to this problem.

```admonish
This is a problem that does not exist in Rust, as there's a single namespace that everyone shares, but that exists in Golang.
The current proposed solution is to introduce an `as` keyword, like in Rust, to be able to alias imports (e.g. `use a::some_lib as a_some_lib;`).
```

## Dependency graph and type checking

During building, a dependency graph of all dependencies is formed (and dependencies are retrieved from Github at the same time). This must be done to detect [dependency cyles](https://en.wikipedia.org/wiki/Circular_dependency).

Once this is done, a list of dependencies from leaves to roots is computed, and each dependency is analyzed in this order.
Dependencies are not compiled! As the circuit-writer is not ran. Things stop at the type checker.
For every new dependency analyzed, all TAST (typed AST) previously computed on previous dependencies are passed as argument.
This way, if a dependency A uses a dependency B, it has access to the TAST of B to perform type checking correctly.

As such, it is important that `a::some_lib` and `b::some_lib` are seen as two independent modules.
For this reason, we store imported modules as their fully qualified path, in the set of TASTs that we pass to the type checker.
But in the current module, we store them as their alias, so that we can use them in the code.

```
TASTs: HashMap<a::some_lib, TAST>
TAST: contains <some_lib -> a::some_lib>
```

## Compilation and circuit generation

Once type checking is done, the circuit writer is given access to all of the dependencies' TAST (which also contain their AST). 
This way, it can jump from AST to AST to generate an unrolled circuit.

# Global environment

In this chapter we will talk about functions.

## Local functions

Third-party libraries can have function names that collide with your own function names. 
Due to this, they are tracked in a different data structure that we will see later.

Local functions include:

* **automatically imported built-ins**. Think functions like `assert` and `assert_eq`. See [here](./basics.md#Builtins-and-use-statements) for a full list.
* **main**, this is the main function that your program runs. Of course if you're writing a library this function is not present.
* **normal functions**, these are functions that you define in your program. They can be recursive.
* **methods**, these are functions that are defined on a type. They can be recursive as well.

Built-ins are different from all other functions listed because they are not written in noname, but written in Rust within the compiler. 

For this reason we track functions according to this enum:

```rust
pub enum FnKind {
    /// signature of the function
    BuiltIn(FnHandle),

    /// Any function declared in the noname program (including main)
    LocalFn(AST)

    /// path, and signature of the function
    Library(Vec<String>),
}

/// An actual handle to the internal function to call to resolve a built-in function call.
pub type FnHandle = fn(&mut CircuitWriter, &[Var], Span) -> Result<Option<Var>>;

pub struct FnInfo {
    pub name: Ident,
    pub sig: FnSig,
    pub kind: FnKind,
    pub span: Span,
}
```

Note that the signature of a `FnHandle` is designed to:

* `&mut CircuitWriter`: take a mutable reference to the circuit writer, this is because built-ins need to be able to register new variables and add gates to the circuit
* `&[Var]`: take an unbounded list of variables, this is because built-ins can take any number of arguments, and different built-ins might take different types of arguments
* `Span`: take a span to return user-friendly errors
* `-> Result<Option<Var>>`: return a `Result` with an `Option` of a `Var`. This is because built-ins can return a variable, or they can return nothing. If they return nothing, then the `Option` will be `None`. If they return a variable, then the `Option` will be `Some(Var)`.

We track all of these functions in the following structure:

```rust
pub struct GlobalEnv {
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    functions: HashMap<String, FnInfo>,

    // ...
}
```

## Handling builtins

Builtins are handled in a special way. They are not written in noname, but in Rust.

## Handling local functions

The parser:

* saves the AST of each function it encounters. Specifically, the function's AST is stored under the `GlobalEnv` (TODO: where exactly?). This is necessary as the circuit writer will have to switch to a function's AST when a function is called (and then return to its own AST).

The first step of the type checker resolves imports by doing the following:

* store all built-ins in the `functions` map of the `GlobalEnv`
* resolve all imports (e.g. `use std::crypto`)
* type check each function individually, and save their signature in the `GlobalEnv` using the `FnSig` type
* type check function calls with the signatures they just saved 

(TODO: this means that function declaration must be ordered. I think it is a GOOD thing)

When a function is called, we do the following:

* if the function is qualified (e.g. `crypto::poseidon`), then lookup imported modules (see next section)
* otherwise, check if the function exists in the `GlobalEnv`, if it doesn't then return an error
* if the function exists, then create a new `FnEnv` and register the arguments as local variables there
* switch to the function's AST and pass the new `FnEnv` as argument
* TODO: how to handle the return value? it should be saved in the `FnEnv`


## Third-party libraries

TODO: write this part

```rust
/// This seems to be used by both the type checker and the AST
// TODO: right now there's only one scope, but if we want to deal with multiple scopes then we'll need to make sure child scopes have access to parent scope, shadowing, etc.
#[derive(Default, Debug)]
pub struct GlobalEnv {
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FuncInScope>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,

    /// the arguments expected by main
    pub main_args: (HashMap<String, FuncArg>, Span),
}

pub type FnHandle = fn(&mut CircuitWriter, &[Var], Span) -> Result<Option<Var>>;

pub enum FuncInScope {
    /// signature of the function
    BuiltIn(FnSig, FnHandle),

    /// path, and signature of the function
    Library(Vec<String>, FnSig),
}
```

```admonish
Not all modules are third-party libraries, some are also built-ins (e.g. `std::crypto`).
```

As part of resolving imports, the type checker looks at third-party libraries differently...

TODO: implement this

TODO: how to handle diamond dependency graph or cycles? We must form a dependency graph first, and resolve dependency according to this graph

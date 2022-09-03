# Type Checker

Noname uses a simple type checker to ensure that all types are consistent in the program.

For example, in code like:

```rust
let x = y + z;
```

the type checker will ensure that `y` and `z` are both field elements (because the operation `+` is used).

And in code like:

```rust
assert_eq(a, b);
```

the type checker will ensure that `a` and `b` are of the same types, since they are being compared.

## Type inference

The type checker can do some simple type inference. For example, in the following code:

```rust
let x = y + z;
```

the type of `x` is inferred to be the same as the type of `y` and `z`.

Inference is willingly kept naive, as more type inference would lead to less readable code.

## Scope

The type checker must be aware of scopes, as it keeps track of the type of variables and functions that are local to each scope.

For example:

```rust
let x = 2;
for i in 0..2 {
    let y = x + 1; // x exists inside of this for loop
}
let z = 1 + y; // BAD: y doesn't exist outside of the for loop
```

To do this, each function is passed an [Environment]() which contains a list of all variables along with their type information.

```rust
pub struct Environment {
    /// created by the type checker, gives a type to every external variable
    pub var_types: HashMap<String, TypeInfo>,

    // ...

    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FuncInScope>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,
}
```

So that shadowing is disallowed, even in different scopes, there is only one variable that is stored, but the scope is stored in `TypeInfo` and matched against the current scope to see if the current scope is the same or a direct child.

An environment is unique to a function, as it is important that different functions can use the same variable names.

Some notes:

* Currently the notion of module is quite shaky. It is used mostly for `crypto::poseidon` at the moment.
* `functions` is mostly used for builtins like `assert_eq`
* `modules` is mostly used for the functions in `std::crypto`, which only contains `crypto::poseidon` atm.

more:

* I think Environment mixes things
* we should be able to create a new Environment whenever we parse a new function, so the functions/modules should be part of another (AvailableToAllScopes)
* variables is something that has nothing to do with the "Type" Environment and should be moved elsewhere no? GateCreationEnv?
* there needs to be a namespace in the typeinfo


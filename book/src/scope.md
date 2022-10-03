# Scope

Like most languages, noname has a notion of scope within a function.
Unlike a lot of languages noname forbids shadowing at all scope level.
This means that eventhough different functions can use local variable with colliding names, the local variable of one function must all have different names.

For example, the following code does not compile:

```rust
let x = 2;
let x = 3; // this won't compile

let y = 4;
for i in 0..4 {
    let y = i; // this won't compile either
}
```

Scopes are only used for:

* for loops
* in the future: if/else statements

## Scope mechanisms

Both the type checker and the circuit writer need to keep track of local variable.
For the type checker (`type_checker.rs`), a `TypeEnv` structure keeps track of the association between all local variables names and their type information.
For the circuit writer (`circuit_writer.rs`), a `FnEnv` structure keeps track of the association between all local variable names and their circuit variable.

Both structure also keep track of how nested the current block is (the top level starting at level 0).
For this reason, it is important to remember to increase the current scope when entering a new block (for loop, if statement, etc.) and to decrease it when exiting the block.
In addition, all variables from a scope must be disabled (but not deleted, in order to detect shadowing) when exiting that scope.

For example, the type checker's `TypeEnv` structure implements the following logic:

```rust
impl TypeEnv {
    // ...


    /// Enters a scoped block.
    pub fn nest(&mut self) {
        self.current_scope += 1;
    }

    /// Exits a scoped block.
    pub fn pop(&mut self) {
        self.current_scope.checked_sub(1).expect("scope bug");

        // disable variables as we exit the scope
        for (name, (scope, type_info)) in self.vars.iter_mut() {
            if *scope > self.current_scope {
                type_info.disabled = true;
            }
        }
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn store_type(&mut self, ident: String, type_info: TypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.clone(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error::new(
                 ErrorKind::DuplicateDefinition(ident),
                 type_info.span,
            )),
            None => Ok(()),
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    pub fn get_type_info(&self, ident: &str) -> Option<TypeInfo> {
        if let Some((scope, type_info)) = self.vars.get(ident) {
            if self.is_in_scope(*scope) && !type_info.disabled {
                Some(type_info.clone())
            } else {
                None
            }
        } else {
            None
        }
    }
}
```

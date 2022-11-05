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

## Another solution

This is a bit annoying. We need a context switcher in both the constraint writer and the type checker, and it's almost the same code.

### Type Checker

### Constraint Writer

```rust
pub struct CircuitWriter {
    /// The type checker state for the main module.
    // Important: this field must not be used directly.
    // This is because, depending on the value of [current_module],
    // the type checker state might be this one, or one of the ones in [dependencies].
    typed: TypeChecker,

    /// The type checker state and source for the dependencies.
    // TODO: perhaps merge {source, typed} in this type?
    dependencies: Dependencies,

    /// The current module. If not set, the main module.
    // Note: this can be an alias that came from a 3rd party library.
    // For example, a 3rd party library might have written `use a::b as c;`.
    // For this reason we must store this as a fully-qualified module.
    pub(crate) current_module: Option<UserRepo>,
```

and then access to the TAST is gated so we can switch context on demand, or figure out what's the current context:

```rust
impl CircuitWriter {
    /// Retrieves the type checker associated to the current module being parsed.
    /// It is possible, when we jump to third-party libraries' code,
    /// that we need access to their type checker state instead of the main module one.
    pub fn current_type_checker(&self) -> &TypeChecker {
        if let Some(current_module) = &self.current_module {
            self.dependencies
                .get_type_checker(current_module)
                .expect(&format!(
                    "bug in the compiler: couldn't find current module: {:?}",
                    current_module
                ))
        } else {
            &self.typed
        }
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.node_types.get(&expr.node_id)
    }

    pub fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.node_types.get(&node_id)
    }

    pub fn struct_info(&self, name: &str) -> Option<&StructInfo> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.struct_info(name)
    }

    pub fn fn_info(&self, name: &str) -> Option<&FnInfo> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.functions.get(name)
    }

    pub fn size_of(&self, typ: &TyKind) -> Result<usize> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.size_of(&self.dependencies, typ)
    }

    pub fn resolve_module(&self, module: &Ident) -> Result<&UsePath> {
        let curr_type_checker = self.current_type_checker();

        let res = curr_type_checker.modules.get(&module.value).ok_or_else(|| {
            self.error(
                ErrorKind::UndefinedModule(module.value.clone()),
                module.span,
            )
        });

        res
    }

    pub fn do_in_submodule<T, F>(&mut self, module: &Option<Ident>, mut closure: F) -> Result<T>
    where
        F: FnMut(&mut CircuitWriter) -> Result<T>,
    {
        if let Some(module) = module {
            let prev_current_module = self.current_module.clone();
            let submodule = self.resolve_module(module)?;
            self.current_module = Some(submodule.into());
            let res = closure(self);
            self.current_module = prev_current_module;
            res
        } else {
            closure(self)
        }
    }

    pub fn get_fn(&self, module: &Option<Ident>, fn_name: &Ident) -> Result<FnInfo> {
        if let Some(module) = module {
            // we may be parsing a function from a 3rd-party library
            // which might also come from another 3rd-party library
            let module = self.resolve_module(module)?;
            self.dependencies.get_fn(module, fn_name) // TODO: add source
        } else {
            let curr_type_checker = self.current_type_checker();
            let fn_info = curr_type_checker
                .functions
                .get(&fn_name.value)
                .cloned()
                .ok_or_else(|| {
                    self.error(
                        ErrorKind::UndefinedFunction(fn_name.value.clone()),
                        fn_name.span,
                    )
                })?;
            Ok(fn_info)
        }
    }

    pub fn get_struct(&self, module: &Option<Ident>, struct_name: &Ident) -> Result<StructInfo> {
        if let Some(module) = module {
            // we may be parsing a struct from a 3rd-party library
            // which might also come from another 3rd-party library
            let module = self.resolve_module(module)?;
            self.dependencies.get_struct(module, struct_name) // TODO: add source
        } else {
            let curr_type_checker = self.current_type_checker();
            let struct_info = curr_type_checker
                .struct_info(&struct_name.value)
                .ok_or(self.error(
                    ErrorKind::UndefinedStruct(struct_name.value.clone()),
                    struct_name.span,
                ))?
                .clone();
            Ok(struct_info)
        }
    }

    pub fn get_source(&self, module: &Option<UserRepo>) -> &str {
        if let Some(module) = module {
            &self
                .dependencies
                .get_type_checker(module)
                .expect(&format!(
                    "type checker bug: can't find current module's (`{module:?}`) file"
                ))
                .src
        } else {
            &self.typed.src
        }
    }

    pub fn get_file(&self, module: &Option<UserRepo>) -> &str {
        if let Some(module) = module {
            &self.dependencies.get_file(module).expect(&format!(
                "type checker bug: can't find current module's (`{module:?}`) file"
            ))
        } else {
            &self.typed.filename
        }
    }

    pub fn get_current_source(&self) -> &str {
        self.get_source(&self.current_module)
    }

    pub fn get_current_file(&self) -> &str {
        self.get_file(&self.current_module)
    }

    pub fn add_local_var(&self, fn_env: &mut FnEnv, var_name: String, var_info: VarInfo) {
        // check for consts first
        let type_checker = self.current_type_checker();
        if let Some(_cst_info) = type_checker.constants.get(&var_name) {
            panic!(
                "type checker bug: we already have a constant with the same name (`{var_name}`)!"
            );
        }

        //
        fn_env.add_local_var(var_name, var_info)
    }

    pub fn get_local_var(&self, fn_env: &FnEnv, var_name: &str) -> VarInfo {
        // check for consts first
        let type_checker = self.current_type_checker();
        if let Some(cst_info) = type_checker.constants.get(var_name) {
            let var = Var::new_constant(cst_info.value, cst_info.typ.span);
            return VarInfo::new(var, false, Some(TyKind::Field));
        }

        // then check for local variables
        fn_env.get_local_var(var_name)
    }
```

we basically have to implement the same in the type checker... It always sort of looks the same. A handy function is either called with `get_fn` or `expr_type` or `node_type` etc. or we call a block of code with `do_in_submodule`.

all of these basically start by figuring out the `curr_type_checker`:

- what's the current module (`self.current_module`)?
  - if there is none, use the main TAST (`self.typed`)
  - otherwise find that TAST (in `self.dependencies`)
  - btw all of this logic is implemented in `self.current_type_checker()`
  - the returned TAST is called `curr_type_checker`

then, if we're handling something that has a module:

- do name resolution (implemented in `resolve_module()`):
  - use `curr_type_checker` to resolve the fully-qualified module name

or if we're executing a block within a module:

- save the current module (`self.current_module`)
- replace it with the module we're using (we have used `resolve_module()` at this point)
- execute in the closure where `self` is passed
- when we return, reset the current module to its previous saved state

# Structs

User can define custom structs like so:

```rust
struct Thing {
    x: Field,
    y: Field,
}
```

and can declare and access such structs like so:

```rust
let thing = Thing { x: 1, y: 2 };
let z = thing.x + thing.y;
```

## implementation

A struct is a list of fields that each have a name and a type.
Like any type, a struct can be serialized as a list of `CellVar`s.
So that fields of an instantiated struct can be accessed, the following information needs to be known:

* at what offset each field is stored
* how many `CellVar` each field takes

During type checking, the type checker calculates the length of each field, and stores this information in the `TypeEnv` structure:

```rust
pub struct FieldInfo {
    typ: TyKind,
    offset: usize,
    len: usize,
}

pub struct TypeEnv {
    // ...

    // struct name -> Vec<(field name -> (type, offset, length))>
    pub structs: HashMap<String, Vec<name, FieldInfo>>,
}
```

The length, in addition to the offset, is important because the circuit writer does not know what types it is really querying (it does not have this information anymore).

--

or:

```rust
let thing = Thing { x: 5, y: 5 };
```

could be stored as thing -> x: var, y: var

and

```rust
thing.x
```

could be retrieved via the hashmap:

```rust
pub enum VarChoice {
    /// Used by simple types
    Flat(Var),

    /// Used by types with fields
    Named(HashMap<String, Var>),

    /// Used by arrays, tuples
    List(Vec<Var>),
}

pub struct VarInfo {
    scope: usize,
    cvars: VarChoice,
}
```

the problem with this is that we can't use assert_eq to check an array or a struct, unless we generalize assert_eq to take a VarChoice and handle all cases. It's easier if everything's just a list of cellvars eventually

it's easy to handle arrays with arbitrary types with the the flat way as well `a: [3; T]` and `a[1]` will just have to figure out the array type

so maybe we should have a hashmap that stores the length of everything, from struct fields to array item

hummm, but the problem is that in `a[1]` or `a.x` the circuit writer doesn't know the type of `a` anymore, and so they don't know what to look for?

solution3: perhaps a var can contain this information themselves? (name -> offset and length, and array could use "array" as name?)

so to recap:

* solution 1: store it in type_env 
  * circuit writer can't access the type of a var anymore
  * one solution would be to create node id like in Rust
  * and store some information under a hashmap linked to a specific node
* solution 2: store it in local_env
  * we have to store that information at creation
  * when we create an array/struct, we have that information
  * although, what about array of array, and struct of struct...
* solution 3: store it in Var
  * but what about struct of struct, and array of array?

```
x.y.z 
```

* I need to know what y is in x so that I can get z


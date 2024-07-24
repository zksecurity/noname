# RFC-0: Generic Parameters

## Summary
This RFC proposes to support generic parameters in noname. The generic parameters can be inferred from the observed arguments, such as constants, arrays, or structs. This improves reusability and modularity of the code. It is a prerequisite for supporting generic types, such as array with symbolic size.

## Code Examples
Here is a few previews of how the generic parameters can be used, and what features it would unlock.

Allow functions to create array with symbolic size:
```rust
const c_mul = 2;

// the return type is determined by the generic arguments
// so `init_arr` can be used to create arrays with different sizes
fn init_arr(const N, const M) -> [Field; N * 2 + M] {
    let arr = [0; N * c_mul + M];
    return arr;
}


fn main() -> [Field; 3 * c_mul + 1] {
    let arr = init_arr(3, 1);
    return arr;
}
```

Inferring the generic values from the observed array argument:
```rust
// N can be inferred from the array size of argument `arr`
fn last(arr: [Field; N]) -> Field {
    // use generic parameter N to form dynamic expressions in the function scope
    return arr[N - 1];
}

fn main() -> Field {
    let arr = [1, 2, 3, 4, 5];
    return last(arr);
}

```

Inferring the generic values from the observed struct argument:
```rust
...

struct House<G> {
   rooms: [Room; G],
}

// it can also infer N and M from the arguments
fn all_rooms(house1: House<N>, house2: House<M>) -> [Room; N + M] {
    // declare default array with symbolic size represented by generic parameters
    let rooms = [Room; N + M];

    // use generic parameters to express for loop range
    for i in 0..N {
        rooms[i] = house1.rooms[i];
    }
    for i in 0..M {
        // use generic parameters to access the array index
        rooms[N + i] = house2.rooms[i];
    }
    return rooms;
}

fn main() -> [Room; 4] {
    let house1 = House {
        rooms: [Room { ... }, Room { ... }, Room { ... }],
    };
    let house2 = House {
        rooms: [Room { ... }],
    };
    // it will type check the inferred type is [Room; 4]
    return all_rooms(house1, house2);
}
```

## Builtin Examples
Given the following function signatures for builtin functions:
```rust
fn from_bits(const N: Field, val: Field) -> [Field; N]
fn from_bits(bits: [Bool; N]) -> Field
```

Calling the builtin functions in native code:
```rust
use std::to_bits;
use std::from_bits;

const num_bits = 8;

fn main() {
    let val1 = 101;
    // `num_bits` will be assigned to the generic parameter `N`
    // then the type of `bits` will be monomorphized to [Bool; 8]
    let bits = to_bits(num_bits, val); 
    // the value of `N` can be determined from the size of `bits` during monomorphization
    // so the builtin function knows how many bits to convert
    let val2 = from_bits(bits); 
    assert_eq(val1, val2);
}
```


The values for the generic parameters will be passed to the function via the `generics` argument:

```rust
fn to_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters, 
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // should be two input vars
    assert_eq!(vars.len(), 2);

    // first var is the number of bits

    // but the better practice would be to retrieve the value from the generics
    let num_var = generics.get("N");

    // alternatively, it can be retrieved from the vars, but it is not recommended
    // let num_var = &vars[0];

    // second var is the value to convert
    let val = &vars[1];

    // convert value to bits
    let mut bits = Vec::new();
    let mut lc = 0;
    let e2 = 1;
    for i in 0..num_var {
        // bits[i] = (in >> i) & 1;
        let bit = compiler.backend.and(
            compiler.backend.shr(val_var, i),
            compiler.backend.constant(1),
        );
        bits.push(bit);

        // bits[i] * (bits[i] -1 ) === 0;
        let zero_mul = compiler.backend.mul(bit, compiler.backend.sub(bit, 1));
        compiler.backend.assert_eq_const(zero_mul, 0, span);

        // lc += bits[i] * e2;
        lc = compiler.backend.add(lc, compiler.backend.mul(bit, e2));

        e2 = e2 + e2;
    }

    compiler.backend.assert_var(val, lc, span);

    Ok(Some(bits))
}
```

Even there is no const argument declared like the `to_bits` example, it can still look up the values from the `generics`:
```rust
fn from_bits<B: Backend>(
    compiler: &mut CircuitWriter<B>,
    generics: &GenericParameters,
    vars: &[VarInfo<B::Field, B::Var>],
    span: Span,
) -> Result<Option<Var<B::Field, B::Var>>> {
    // should be one input var
    assert_eq!(vars.len(), 1);

    let bits = &vars[0];

    let num_bits = generics.get("N");

    let mut lc = 0;
    let e2 = 1;
    // elements should be boolean type
    for i in 0..num_bits {
        let bit = &bits[i];

        // lc += bits[i] * e2;
        lc = compiler.backend.add(lc, compiler.backend.mul(bit.value(), e2));
        e2 = e2 + e2;
    }

    Ok(Some(Var::new(lc)))
```

Both the return type and returned vars can be checked outside of the builtin functions. The return type can be checked automatically in the same way as the native functions, the types of which are propagated and converged at certain point, at which error will throw if the types are not matched.

The return vars can be checked by relying on the types. Each concrete type has a fixed number of vars. With the resolved return type of the builtin function, we can check if the size is matched. Additionally, we can check the values recursively with the type structure, but it might only limited to checking the boolean type which got obvious bound 0 or 1.


## monomorphization
The inference of the generic values can be done by observing the arguments passed to the function. Then it stores the inferred values in the relevant contexts for the following compiler pipeline to do type checking and circuit synthesizing. We call the process of inferring the generic values and type checking _Monomorphization_.

The current pipeline of compiling noname code is:
1. Parse the code into AST
2. Convert the AST into NAST with naming resolution
3. Convert the NAST into TAST with type metadata collection and type checking
4. Circuit synthesizing TAST into an arithemtic backend

With generic parameters, the current TAST phase can't handle the type checking anymore, because the generic parameters are unknown. For example, it can't type check the array with symbolic size without inferring for the generic parameters.

To extend the type checking to support generic parameters, we can add a MAST phase (Monomorphized AST), right after the TAST phase, to infer the values of the generic parameters. The circuit synthesizer will rely on the MAST instead of the TAST to compile a circuit.

The pipeline will be: AST Parser -> NAST -> TAST -> MAST -> Circuit synthesizing

## Implementation
To support the generic syntax as shown in the examples above, we need to make changes to the AST Parser support generic syntax. Furthermore, because the generic parameters can't be inferred at TAST phase, the type checking for the concrete types, which is done in `check_block` function, will be deferred to MAST phase.

The newly added phase MAST will be responsible for inferring the generic values from the observed arguments. It will also be responsible for type checking the inferred generic values. Circuit synthesizer will rely on inferred values from MAST and type metadata to determine how to organize the variables at the lowest level, `CellVar`.

### Generic Syntax

This RFC proposes a simple generic syntax without the introduction of the turbofish syntax, since we don't need to infer the generic parameters from the function arguments. Instead, the values of the generic parameters can be directly assigned by comparing values with the observed arguments.

For example, with the turbofish, we could do something like:

```rust
fn create_arr<N>(arr: [Field; N + 3]) -> [Field; N + 3] {...}
```

This is a rare case where the generic parameter can't be trivially inferred from the observed arguments. To get it work without any advanced inference setups, it would require passing the value of N to the function via turbofish syntax, such as:

```rust
// a is then of type [Field, 6]
let a = create_arr::<3>(arr);
```

The turbofish syntax allows direct passing the values for the generic parameter. However, for most of the cases, the values for the generic parameters can be obtained simply by observing the arguments passed to the function. This RFC aims to keep the syntax simple and to be intuitive.

Without the turbofish syntax, the generic syntax can be simplified like the following:

```rust
// the value of N can be equal to the argument directly
fn create_arr(const N) -> [typ; N]

// if the argument is array, then the value of N can be equal to the size of the array
fn last(arr: [typ; N]) -> Field
fn join(arr1: [typ; N], arr2: [typ; M]) -> [typ; N + M]
```

In the function scope, it might need to determine whether a variable is a generic parameter or not. We can reserve the single capital alphabets for the generic parameters. For example, `N`, `M`, `G`, `K`, etc. The generic parameters can be used in the function arguments, return type, or expressions. 

### AST Parser
Parser will need to collect the generic identifiers for the following constructions `FunctionDef`, `StructDef`. It will add two `TyKind`: `ConstGeneric` and `GenericArray`. The size of `GenericArray` can be represented by a `Symbolic` value, which can contain generic parameters.

We add `generics` field to `FunctionDef` and `StructDef`. `generics` is a set of `GenericIdentifier`, which can be just a string.

`GenericArray` with `Symbolic` size:
```rust
enum Symbolic {
    Concrete(u32),
    Generic(GenericIdentifier),
    Add(Symbolic, Symbolic),
    Mul(Symbolic, Symbolic),
}

GenericArray {
    ty: TyKind,
    size: Symbolic,
}
```

Update `FunctionDef`:
```rust
pub struct FunctionDef {
    pub sig: FnSig,
    pub body: Vec<Stmt>,
    pub span: Span,
    // to add
    pub generics: HashSet<GenericIdentifier>,
}
```

Update `StructDef`:
```rust
pub struct StructDef {
    pub module: ModulePath, 
    pub name: CustomType,
    // to add
    pub generic: HashSet<GenericIdentifier>,
    ...
}
```


Example for a function with a generic parameter:
```rust
fn init_arr(n: N) -> [Field; N * 2 + 1] {...}
```

The parser should create the function definition like pseudo code below:

```rust
FunctionDef{
    generics = ["N"],
    FnSig = {
        ...
        FnArg = {
            typ: ConstGeneric("N"), 
            name: 'n', 
        }
        // Add / Mul / Generic / Concrete are variants of Symbolic enum
        return_type: GenericArray(Field, Add(Mul(Generic("N"), Concrete(2)), Concrete(1)))
    }
}
```

Example for a struct with a generic parameter:
```rust
struct House<G> {
   rooms: [Room; G],
}
```

The parser should create the struct definition Like pseudo code below:
```rust
StructDef {
    ...
    generics = ["G"],
    fields = {
        "rooms": GenericArray(Room, Generic("G")),
    }
}
```

The TAST use these metadata addtions of generic parameters for type checking the consistency of generic identifiers. In MAST phase, they will be useful for inferring the generic values from the observed arguments.

### TAST
The consistency of the generic parameters should be checked. For example, the generic identifiers, such as N or G, should exist in the `generics` set of the function or struct definition. Also, all the generic parameters should be used in the function arguments.

*Type check generic parameters for functions*
```rust
// shouldn't allow this, because the N should be defined in the function arguments
fn foo(n: Field) -> [Field; N] {...}

// not allowed if no use of N in the body
fn foo(const N) {...} 
fn foo(arr: [Field; N]) {...} 
fn foo(thing: Thing<N>) {...}
```

The bottom line is that the generic parameters should be used in the function arguments. This is because the inferring algorithm will be based on the observed arguments.

*Type check generic parameters for structs*
```rust
// shouldn't allow unused generic parameters
struct House<N> {
    rooms: [Room; 3],
}

// allowed
struct House<N> {
    rooms: [Room; N],
}
```

*Defer type checks*
Anything involves the generic parameters should be deferred to MAST phase. They can be:
- Array with generic size
- Struct with generic parameters
- Generic parameters in the function arguments

In MAST phase, generic parameters can be inferred to values, so the symbolic values can be evaluated. Thus, all the types with generic parameters can be type checked, as the array sizes become concrete values. 

So we will need to shift the the current `check_block` to MAST phase for type checking.


### MAST
With the type information collected from the TAST phase, the MAST phase can infer the generic values from the observed arguments by running through the `check_block` function, which is used to be done in TAST phase. The `check_block` is context aware, and will recursively walk through the main function body. We can devise an algorithm to incorporate with `check_block` to infer the generic values and store in a certain way for later use.


### Inferring algorithm
The algorithm will need to handle the following inference categories:
- Inferencing from constant argument
- Inferencing from array arugment
- Inferencing from struct argument

Example of inferring constant arguments:
```rust
// constant generic
// - the value of N can be inferred from the observed value of `size`
// - store the value in the context
fn gen_arr(const N) -> [Field; N * 3 + 1] {
    let arr = [Field; N * 3 + 1];
    return arr;
}
```


Example of inferring from array arguments:
```rust
// First, here is a simple case.
// - the N can be inferred from the array size of argument `arr` 
// - store the value of N in the context
fn last(arr: [Field; N]) -> Field {
    return arr[N - 1];
}
```

```rust
// Then, here is a more challenging case that would require techniques like SMT to solve the unique values of the generic parameters.
// Even N * 2 looks obvious to solve, solving it may need something like (computer algebra system) CAS in rust.
// It is easy to evaluate the Symbolic values using the solved generic value. But the way around is difficult.
fn last(arr: [Field; N * 2]) -> Field {
    return arr[N - 1];
}
```

In this RFC, we might just enforce the syntax to be sufficient to support the simple cases. That is to disallow arithemtic operations among generic parameters for function arguments. 


Example of inferring from struct arguments
```rust
struct House<G> {
   id: Field,
   rooms: [Room; G],
}

// struct with generic parameter
// - store the generic parameter mapping in the function context, e.g { G: N }
// - resursively walk through the struct fields
// - infer the generic value from the corresponding observed struct field, which could be from a constant or array size.
// e.g.
// 1. given observed struct argument: ```house = House {rooms: [Room{...}, Room{...}, Room...]}```
// 2. TyKind for `house` is Array(Room, 3)
// 3. the corresponding function argument definition is `GenericArray(Room, G)`
// 4. using the inferring algorithm for generic array, G = 3
// 5. update the function generic parameter N = 3 via the mapping {G: N}
fn double(house: House<N>) -> House<2 * N> {
    // init house with doubled size of rooms
    // the expression N * 2 is evaluated to 6, 
    let new_house = House {
        id: house.id,
        rooms: [Room; N * 2],
    };

    return new_house;
}
```


To recap, here is the pseudo code for the inferring algorithm for function
```rust
// at function caller scope
// collect observed arguments
//  at function callee scope
//    for each argument
//      if the argument involves generic parameters
//        infer the generic values from observed arguments
//    for each statement
//      compute the type with the inferred generic values
//      type checks
//  return type
```

### Type Check Algorithm 

Pseudo code for the type check algorithm in MAST phase:
```rust
// load its ExprKind
// if it is a generic type, such as GenericArray or ConstGeneric
//    load the generic values from the context
//    evaluate the Generic type to concrete type, such as Array or Constant
// type check with the expected type
```

Reusing the previous example in the inferring algorithm section, the type checking would be like:
```rust
struct House<G> {
   rooms: [Room; G],
}

fn double(house: House<N>) -> House<2 * N> {
    let new_house = House {
        id: house.id,
        rooms: [Room; N * 2],
    };

    // type check the return value
    // compute the type of new_house in generic form House<6>
    // compare with the return type House<2 * N>
    return new_house;

    // we might need to update the TyKind::Custom to have a mapping of generic parameters to concrete values.
    // so `new_house` => Custom{generic: {"G": 6}}, 
    // while the return type can be also computed as Custom{generic: {"G": 2 * N}} 
    // where it uses the inferred N = 3 in the scope.
}
```

Example for generic array:
```rust
// both N and M are inferred from the observed arguments house1 and house2 passed from main function
fn all_rooms(house1: House<N>, house2: House<M>) -> [Room; N + M] {
    // `compute_type` steps into the case for `ExprKind::GenericArray`
    // load the generic values N and M from the context
    // convert the GenericArray to concrete Array
    // store the concrete Array in MastEnv using expr.node_id as key
    let rooms = [Room; N + M];
    for i in 0..N {
        rooms[i] = house1.rooms[i];
    }
    for i in 0..M {
        rooms[N + i] = house2.rooms[i];
    }

    // should type check with the return type [Room; N + M] as the concrete type eg. [Room; 4]
    // rooms = TyKind::Array(Room, 4); while the return type is TyKind::Array(Room, N + M) where N = 3, M = 1
    return rooms;
}

fn main() {
    ...
    // type of house1 is Array(Room, 3)
    // type of house2 is Array(Room, 1)
    let rooms = all_rooms(house1, house2);
    ...
}
```

The current `compute_type` function can remain as it is. It stores the computed concrete types, such as `Array` or `Custom` as shown in the examples above, instead of the generic types. It creates a mapping between the expression node from the AST and the type `TypeKind`.

### Circuit Synthesizer
Circuit synthesizer relies on the `compute_expr` function to walk through the AST, which is the `FunctionDef`as the entry point. Using the mapping between expression node and the type, the results of the lookup will be always in concrete types. 

The current `size_of` function of the `TypeChecker` is to determine the size of a type. It could do so when the declarations of the types are without generic parameters. With the generic parameters, the `size_of` function will need to be refactored to look up the concrete type via express node id.

Pseudo code of refactored `size_of` function:

```rust
pub(crate) fn size_of(&self, typ: &TyKind) -> usize {
    match typ {
        TyKind::Field => 1,
        TyKind::Custom { module, name, generics } => {
            let qualified = FullyQualified::new(&module, &name);
            let struct_info = self
                .struct_info(&qualified)
                .expect("bug in the type checker: cannot find struct info");

            // evaluate the size using the inferred generic values
            // internally, it will recursively call size_of to sum up size of the struct fields
            struct_info.size(generics)
        }
        TyKind::BigInt => 1,
        TyKind::Array(typ, len) => (*len as usize) * self.size_of(typ),
        TyKind::Bool => 1,
    }
}
```

## Alternative approach
[One alternative approach](https://github.com/zksecurity/noname/pull/136) to the monomorphization described above is to propagate the generic values directly in circuit writer, without the need to add the MAST phase.

The circuit writer walks through the original AST via the `compile_expr` function. This function propagate the values from the main function argument and constants and compute the `VarOrRef` as an result. The `VarOrRef` doesn't return the struture of the types being computed.

In the process, when it needs to determine the structure of the type behind an expression node, it relies on the `size_of` function to determine the number of vars representing the type. The `size_of` relies on the node id of an expression to look up the type. This is not a problem when the types are concrete.

When the type behind an expression node is generic, the way of looking up the size of a type via `size_of` is not applicable anymore, since the expression node can be of a generic type.

To solve this problem, there should be a new way to determine the size of a type for an expression node without relyin on the node id. One way, described `ComputedExpr`, is to retain the structure of the type through the propagation in `compute_expr`. Instead of passing around the `VarOrRef`, the `compute_expr` returns `ComputedExpr` which contains both the structure of the type and the underlying variables `VarOrRef`.

For example, when it is computing for the `ExprKind::ArrayAccess`, it can use the `ComputedExpr` of the `array` expression node to determine the size of the array, so as to do some bound checks for access index.

This approach would require a significant refactor of the circuit writer's compilation process. It would require changes to the assumptions from using `VarOrRef` to structured `ComputedExpr`. It would also need to rely on `ComputedExpr` to do some addtional checks instead of just relying on types. This would require quite a number of additional assumptions between the `ComputedExpr`, the actual types and generic parameters.

Therefore, we thought the monomorphization approach is more straightforward and easier to maintain in a long run, considering the pipeline of the compiler.

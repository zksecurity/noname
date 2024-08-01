# RFC-0: Generic-sized arrays in function signatures

## Summary
This RFC proposes to support const generics in noname. The generic parameters can be resolved from the observed arguments, such as constants, arrays, or structs. This improves reusability and modularity of the code. It is a prerequisite for supporting generic array with symbolic size.

## Code Examples
Here is a few previews of how the generic parameters can be used, and what features it would unlock.

Allow functions to create array with symbolic size:
```rust
// the return type is determined by the generic arguments
// so `init_arr` can be used to create arrays with different sizes
fn init_arr(const LEN: Field) -> [Field; LEN] {
    let arr = [0; LEN];
    return arr;
}


fn main() -> [Field; 3] {
    let arr = init_arr(3);
    return arr;
}
```

Resolving the generic values from the observed array argument:
```rust
fn last(arr: [Field; LEN]) -> Field {
    // use generic parameter LEN to form dynamic expressions in the function scope
    return arr[LEN - 1];
}

fn main() -> Field {
    let arr = [1, 2, 3, 4, 5];
    // generic parameter LEN can be resolved from the array size of argument `arr`
    return last(arr);
}

```


## Builtin Examples
Given the following function signatures for builtin functions:
```rust
fn to_bits(const LEN: Field, val: Field) -> [Field; LEN]
fn from_bits(bits: [Bool; LEN]) -> Field
```

Calling the builtin functions in native code:
```rust
use std::bits;

const num_bits = 8;

fn main() {
    let val1 = 101;
    // `num_bits` will be assigned to the generic parameter `LEN` in the return type
    // then the type of `bits` will be monomorphized to [Bool; 8]
    let bits = bits::to_bits(num_bits, val); 
    // the value of `LEN` can be determined from the size of `bits` during monomorphization
    // so the builtin function knows how many bits to convert
    let val2 = bits::from_bits(bits); 
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
    ...

    // retrieve the generic values from the `generics` argument
    let bitlen = generics.get("LEN") as usize;

    ...
}
```

Both the return type and returned vars can be checked outside of the builtin functions. The return type can be checked automatically in the same way as the native functions, the types of which are propagated and converged at certain point, at which error will be thrown if the types are not matched.

The return vars can be checked by relying on the types. Each concrete type has a fixed number of vars. With the resolved return type of the builtin function, we can check if the size is matched. Additionally, we can check the values recursively with the type structure, but it might only limited to checking the boolean type which got obvious bound 0 or 1. So automatically checking if the actual return from a builtin is an area to be improved in the future.


## monomorphization
The resolving of the generic values can be done by observing the arguments passed to the function. Then it stores the resolved values in the relevant contexts for the following compiler pipeline to do type checking and circuit synthesizing. We call the process of resolving the generic values and type checking _Monomorphization_.

The current pipeline of compiling noname code is:
1. Parse the code into AST
2. Convert the AST into NAST with naming resolution
3. Convert the NAST into TAST with type metadata collection and type checking
4. Circuit synthesizing TAST into an arithemtic backend

With generic parameters, the current TAST phase can't handle the type checking anymore, because the generic parameters are unknown. For example, it can't type check the array with symbolic size without resolving the values for the generic parameters.

To extend the type checking to support generic parameters, we can add a MAST phase (Monomorphized AST), right after the TAST phase, to resolve the values of the generic parameters. The circuit synthesizer will rely on the MAST instead of the TAST to compile a circuit.

The new pipeline will be: AST Parser -> NAST -> TAST -> MAST -> Circuit synthesizing

## Implementation
To support the generic syntax as shown in the examples above, we need to make changes to the AST Parser support generic syntax. Furthermore, because the generic parameters can't be resolved at TAST phase, the some type checkings will be less strict and deferred to MAST phase.

The newly added phase MAST will be responsible for resolving the generic values from the observed arguments. It includes type checking on the monomorphized types that are bypass in the TAST phase. 

### Generic Syntax

This RFC proposes a simple generic syntax without the introduction of the common turbofish syntax, since we don't need to resolve the generic parameters from the function arguments. Instead, the values of the generic parameters can be directly resolved by comparing values with the observed arguments.

For example, with the turbofish, we could do something like:

```rust
fn create_arr<N>(arr: [Field; N + 3]) -> [Field; N + 3] {...}
```

This is a rare case where the generic parameter can't be trivially resolved from the observed arguments. To get it work without any advanced inference setups, it would require manually passing the value of `N` to the function via turbofish syntax, such as:

```rust
// a is then of type [Field, 6]
let a = create_arr::<3>(arr);
```

However, for most of the cases, the values for the generic parameters can be obtained simply by observing the arguments passed to the function. This RFC aims to keep the syntax simple and to be intuitive. Without the turbofish syntax, the generic syntax can be simplified like the following:

```rust
// the value of LEN equals to the argument passed in
fn create_arr(const LEN: Field) -> [typ; LEN]

// if the argument is array, then the value of LEN equals to the size of the array
fn last(arr: [typ; LEN]) -> Field
```

In the function scope, it might need to determine whether a variable is a generic parameter or not. We rules strings with at least 2 letters, which should be all capitalized, as generic parameters. 

### AST Parser
Parser will need to collect the generic identifiers for the following constructions `FunctionDef`. It will add a new `TyKind`, the `GenericSizedArray(type, size)`. The size of `GenericSizedArray` is represented by a `Symbolic` value, which can contain generic parameters or concrete values.

We add `generics` field to `FunctionDef`, which is a set of `GenericParameters` mapping between generic names and values.


```rust
enum Symbolic {
    Concrete(u32), // literal value
    Generic(GenericParameters), // generic parameters
    Constant(Ident), // pointing to a constant variable
}

GenericSizedArray {
    ty: TyKind,
    size: Symbolic,
}
```

Update `FunctionDef`:
```rust
pub struct FunctionDef {
    ...
    // to add
    pub generics: HashSet<GenericIdentifier>,
}
```

Example for a function with a generic parameter:
```rust
fn create_arr(const LEN: Field) -> [Field; LEN] {...}
```

The parser should create the function definition like pseudo code below:

```rust
FunctionDef{
    FnSig = {
        ...
        generics = {"LEN": null},
        FnArg = {
            name: 'LEN', 
        }
        // Add / Mul / Generic / Concrete are variants of Symbolic enum
        return_type: GenericSizedArray(Field, Generic("LEN"))
    }
}
```

The TAST use these metadata of generic parameters for type checking the consistency of generic identifiers. In MAST phase, they will be useful for resolving the generic values from the observed arguments.

### TAST
The generic values are resolved from the observed arguments. If the generic parameters are declared, they should be used in the function body. We need to check if the generic parameters declared make senses.

*Type check generic parameters for functions*
```rust
// shouldn't allow this, because the LEN should be defined in the function arguments
fn foo(n: Field) -> [Field; LEN] {...}

// not allowed if no use of NN in the body
fn foo(const NN: Field) {...} 
fn foo(arr: [Field; NN]) {...} 
```

*Forbid generic function in for-loop*
```rust
for ii in 0..NN {
    // any function takes the for loop var as its argument should be forbidden
    fn_call(ii);
}
```

To allow generic functions in for-loop, we will need to take care of unrolling the loop and instantiating the function with the concrete value of the loop variable. This is not in the scope of this RFC.

*Forbid operations on symbolic value of arguments*
```rust
// disallow arithmetic operations on the symbolic value of the function arguments,
// such as NN * 2 in this case.
// because it is challenging to resolve the value of NN.
fn last(arr: [Field; NN * 2]) -> Field {
    return arr[NN - 1];
}
```

*Defer type checks*
Anything involves the generic parameters should be deferred to MAST phase. We need to defer the type checks for array with generic size.

In MAST phase, the values of generic parameters can be resolved, so the symbolic values can be evaluated. Thus, all the types with generic parameters can be type checked, as the array sizes become concrete values. 

### MAST
After the TAST phase, the MAST phase can resolve the generic values from the observed arguments by propagate the constant values through the main function AST.

### Resolving algorithm
The algorithm will need to handle the following two categories:
- Resolving from constant argument
- resolving from array argument

Example of resolving constant arguments:
```rust
// constant generic
// - the value of LEN can be resolved from an observed constant value propagated
// - store the value in the function body scope
fn gen_arr(const LEN: Field) -> [Field; LEN] {
    let arr = [Field; LEN];
    return arr;
}
```


Example of resolving from array arguments:
```rust
// First, here is a simple case.
// - the LEN can be resolved from the array size of argument `arr` 
// - store the value of N in the context
fn last(arr: [Field; LEN]) -> Field {
    return arr[LEN - 1];
}
```

```rust
// Then, here is a more challenging case that would require techniques like SMT to solve the unique values of the generic parameters.
// Even LEN * 2 looks obvious to solve, solving it may need something like (computer algebra system) CAS in rust.
// It is easy to evaluate the Symbolic values using the solved generic value. But the way around is difficult.
fn last(arr: [Field; LEN * 2]) -> Field {
    return arr[LEN - 1];
}
```

In this RFC, we want to just enforce the syntax to be sufficient to support the simple cases. That is to disallow arithemtic operations among generic parameters for function arguments. 


To recap, here is the pseudo code for the resolving algorithm for function
```rust
// at function caller scope
// collect observed arguments
//  at function callee scope
//    for each argument
//      if the argument involves generic parameters
//        resolve the generic values from observed arguments
//    for each statement
//      compute the type with the resolved generic values
//      type checks
//  return type
```

### Function Call Instantiation
The functions are defined as `FunctionDef`, which is an AST containing the signature and the body of the function. The body is a vector of statements, each of which is a tree of expression nodes. It is fine to have different function calls pointing to these functions' original AST, when the content of these functions doesn't change, and so are the expression nodes.

However, when a function takes generic arguments, the actual arguments can result in different expression nodes. The two calls should point to two different monomorphized function instances. For example:

```rust
fn last(arr: [Field; LEN]) -> Field {
    return arr[LEN - 1];
}

fn main() {
    let arr1 = [1, 2, 3, 4, 5];
    let arr2 = [6, 7, 8, 9];

    let last1 = last(arr1); // with LEN = 5
    let last2 = last(arr2); // with LEN = 4
}
```

The monomorphized body of the function call for `last(arr1)` is `return arr[5 - 1]`, while the one for `last(arr2)` is `return arr[4 - 1]`. Therefore, we can't have a single expression node representing both `arr[5 - 1]` and `arr[4 - 1]` expression nodes. These functions should be instantiated with new ASTs, which are monomorphized from the original ASTs. They will be regenerated with the generic parameters being resolved with concrete values. 

To ensure no conflicts in the node IDs being regenerated for these instantiated functions, the AST for the main function as an entry point will be regenerated. The monomorphized AST preserves the span information to point to the noname source code for the existing debugging feature.

Same as before, these instantiated functions can be pointed by the expression nodes `ExprKind::FnCall`. With the support of generic parameters, we need to change the way of loading the function AST, as the current fully qualified name pattern doesn't contain the information to differentiate the instantiated functions with different generic values.

Thus we can generate the monomorphized function name, and use it to store the monomorphized function AST instead of the original function name. The new string pattern to store the monomorphized function AST can be:
`fn_full_qualified_name#generic1=value1#generic2=value2`


*Type checking*
The instantiation of a generic function will resolve the generic types to be concrete types. Similar to the TAST phase, during the monomorphization of a function body, the computed concrete type can be propagated and compared with the return type of the function signature. 

The type check in this phase will always be in concrete type. Any unresolved generic type will fail the type check.

### Monomorphization Process
Here is an overview of the monomorphization process:

1. Propagates types in the same way the type checker was doing but also with constant values, which will be used to resolve the generic parameters.

2. Along the way, it also propagates the AST nodes. When it is not part of a generic function, the node should remain the same. Otherwise, the node should be regenerated.

3. Whenever it encounters a generic function call, it instantiates the function based on the arguments and store it as a new function with a monomorphized name, then walks through the instantiated function AST. The function call AST node will be modified with the monomorphized name while retaining the same node id and span.

4. In the function instantiation process, all the AST nodes will be regenerated. This new AST will be stored under the monomorphized function name.

5. After monomorphized a function, it should add the name of the original function to a list that records which function AST to delete at the end. We can't not delete the original function AST immediately, because it might be called at different places.

6. In each function block scope, it should type check the return types, by comparing the propagated return type and the defined return type. All these types should be in concrete form without generic parameters involved.

7. At the end, it overrides the main function AST with the monomorphized version, and delete generic functions based on the list.


### Circuit Synthesizer
Circuit synthesizer will rely on the monomorphized AST to compile the circuit. To synthesizer, the workflow will be the same as before, but with the monomorphized AST. It doesn't need to be aware of the newly added support related to generics. The added MAST phase simplifies what needs to be done in the circuit synthesizer to support the generic features, in comparison to the alternative approach described in the following section.

## Alternative approach
[One alternative approach](https://github.com/zksecurity/noname/pull/136) to the monomorphization described above is to propagate the generic values directly in circuit writer, without the need to add the MAST phase.

The circuit writer walks through the original AST via the `compile_expr` function. This function propagate the values from the main function argument and constants and compute the `VarOrRef` as an result. The `VarOrRef` doesn't return the struture of the types being computed.

In the process, when it needs to determine the structure of the type behind an expression node, it relies on the `size_of` function to determine the number of vars representing the type. The `size_of` relies on the node id of an expression to look up the type. This is not a problem when the types are concrete.

When the type behind an expression node is generic, the way of looking up the size of a type via `size_of` is not applicable anymore, since the expression node can be of a generic type.

To solve this problem, there should be a new way to determine the size of a type for an expression node without relyin on the node id. One way, described `ComputedExpr`, is to retain the structure of the type through the propagation in `compute_expr`. Instead of passing around the `VarOrRef`, the `compute_expr` returns `ComputedExpr` which contains both the structure of the type and the underlying variables `VarOrRef`.

For example, when it is computing for the `ExprKind::ArrayAccess`, it can use the `ComputedExpr` of the `array` expression node to determine the size of the array, so as to do some bound checks for access index.

This approach would require a significant refactor of the circuit writer's compilation process. It would require changes to the assumptions from using `VarOrRef` to structured `ComputedExpr`. It would also need to rely on `ComputedExpr` to do some addtional checks instead of just relying on types. This would require quite a number of additional assumptions between the `ComputedExpr`, the actual types and generic parameters.

Therefore, we thought the monomorphization approach is more straightforward and easier to maintain in a long run, considering the pipeline of the compiler.

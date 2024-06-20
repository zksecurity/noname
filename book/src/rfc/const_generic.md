## Summary
Allow array declaration size to be generic on const variable.

## Motivation
Currently, we can only initialize array by specifying the actual values as its elements. To create an array with thounsands of elements, we have to manually write down all the values. 

Without being able to use const variable as array size, it is impossible to create a function that is generic on the size of the returned array. Also the for loop has to be hardcoded on the size of the array, which is not very flexible.

With the support of const variable as array size, the functions can be more reusable and significantly improve modularization.


## detailed design

### declare a const argument

```rust
fn const_generic(const cst: Field) -> [Field; cst] {
    let mut xx = [1; cst];
    for ii in 1..cst {
        xx[ii] = 2;
    }
    return xx;
}
```

### applying a const as a argument

```rust
const cst = 300;
fn const_generic(yy: Field) -> [Field; cst] {
    let xx = [1; cst];
    return xx;
}
```

### initiate array with size of const variable

```rust
const rooms = 3;
const room_size = 20;

struct Room {
    size: Field,
}

struct House {
    rooms: [Room; rooms],
}

fn build_house(const cst: Field) -> [House; cst] {
    let houses = [
        House {
            rooms: [Room {size: room_size}; rooms]
        }; 
        cst
    ];
    return houses;
}
```


## type checking const variable

The current type checker is designed to do local type checks. For the cases of const variable as array size, the type checker has to trace the values globally. This is out of the scope for the design of the current type checker.

Instead of checking the const variable in the type checker, we can check it in the circuit generation phase. 

### changes from previous version

In current circuit generation phase, it computes the low level variables `cvars` via the NAST. It starts with the main function and recursively allocates `cvars` to different variables `VarOrRef`. These `VarOrRef` are passing around in the circuit generation process for other different places to further calculate.

```rust
houses[1].rooms[2].size
```

The variable `houses` holds all the `cvars` to represent all the data behind the variable. If `houses` is a mutable variable, it will be a reference, which records the following instead of holding the actual `cvars` array:
 - the variable name `houses` in local scope
 - start index of the `cvars` to represent the current access path
 - the length of the `cvars` to represent size of the element targeted by the current access path

So when it wants to update the elements inside the `houses`, it will need to reference the start index and the length of the `cvars` of the elements, so that it can narrow down the scope of the `cvars` to be updated.

The following code is how it narrow down the scope for a field access or array access:

```rust
pub(crate) fn narrow(&self, start: usize, len: usize) -> Self {
    match self {
        VarOrRef::Var(var) => {
            let cvars = var.range(start, len).to_vec();
            VarOrRef::Var(Var::new(cvars, var.span))
        }

        //      old_start
        //      |
        //      v
        // |----[-----------]-----| <-- var.cvars
        //       <--------->
        //         old_len
        //
        //
        //          start
        //          |
        //          v
        //      |---[-----]-|
        //           <--->
        //            len
        //
        VarOrRef::Ref {
            var_name,
            start: old_start,
            len: old_len,
        } => {
            // ensure that the new range is contained in the older range
            assert!(start < *old_len); // lower bound
            assert!(start + len <= *old_len); // upper bound
            assert!(len > 0); // empty range not allowed

            Self::Ref {
                var_name: var_name.clone(),
                start: old_start + start,
                len,
            }
        }
    }
}
```

This allows to find the range of the `cvars` belonging to the element targeted by the current access path under the local variable behind the `var_name`, which in this case is the `houses`. 

But it doesn't keep track of the structure of the data. For example, when we are accessing `houses[1].rooms[2]`, it should be able to check if the access is within the array bound. Although the current design can check if it is out of bound by checking access index against the total number of `cvars` for `houses`, it can't label which element access is out of bound as it could be either out of bound of `houses[1]` or the nested access `rooms[2]`.

### Introduce ComputedExpr

With the const variable as array size, we need to propagate its value in circuit generation phase to check if the access is within the bound of the array. So we need a way to retain structure information of a variable that is currently represented by `VarOrRef`.

We can add a layer of abstraction to the `VarOrRef` to keep track of the structure of the data. We can call it `ComputedExpr`. It is computed from `compute_expr` function, which recursively calculates the values (`cvars`) behind the `Expr`. Then the `ComputedExpr` keeps track of the structure of the data, and being passed around.

With the `ComputedExpr` as the result of `compute_expr` calls, the field / array access can determine the structure of the value with actual size of the current variable. For example, when we are accessing `houses[1].rooms[2]`, it can check if the access is within the bound of `houses` and `rooms`, by checking the structure and size of a `ComputedExpr` representing the targeted access path.

In simple view, `ComputedExpr` can be seen as an `Expr` with const variables folded into actual values. 

### Assignment using ComputedExpr

Instead of using start index and length to narrow down the scope of the `cvars` to be updated, we can use the `ComputedExpr` to represent the access path, which iteratively determined when calculating for `Expr::FieldAccess` and `Expr::ArrayAccess`.

When it comes to the `Expr::Assignment`, it can use the access path embeded from a `ComputedExpr` to locate a nested `ComputedExpr` to swap with a new `ComputedExpr`. Then it saves the VarInfo with same name `houses` as local variable but holding the updated tree of `ComputedExpr` that represents a new set of `cvars`.

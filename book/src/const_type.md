# Constants

There's different types of constants:

on the type-checker side:

* top-level variables declared with the keyword `const`. This means that the value passed must be known at compiled time (literals). They have a type that is 
* literals, which have type (`TyKind`) `BigInt`

on the circuit-writer side:

* circuit variables, which can be constants or not

## Importance of the distinction

on the type-checker side:

* some functions take `const` arguments
* array indexes require the value to be a constant

on the circuit-writer side:

* constants are handled differently from non-constants for optimizations purposes

## Problem

First, the problem we're having is a type-checker issue.

The biggest problem we're facing, is a field in the type system can be represented in `TyKind` as `Field` or `BigInt`, which makes type checking harder to check.

Ideally, both types should be merged as `Field`, and constants should be a property of the variable, in addition to the type. 

`TypeInfo` seems to be the right thing to use?

* `TypeInfo` knows if an argument of a function is a constant due to the `const` attribute that a user can use.
* For variables inside of a function. TKTK
* For top-level constants. TKTK

## Implementation detail

The current way constants are implemented is messy.

* `TyKind` has a type `BigInt` that is set when a literal is written in the code
* `AttributeKind` has a tag called `Const` set when the `const` keyword is used before the name of an argument in a function
* `TypeInfo` has a `constant` field that gets set when the variable has an attribute set to `Const` (previous bullet point)

## Solution

use typeinfo everywhere?

but typeinfo has things like `span` that don't necessarily make sense to carry, and `disabled` that's only an `env` thing.

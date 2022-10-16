# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- fix negated (e.g. `-5`) and not (e.g. `!true`) expressions
- bugfix: arguments to the main functions are now constrained
- enforce parenthesis when chaining arithmetic operations
- fix: return an error when a return statement is missing

## [0.6.0] - 2022-10-14

- if/else expressions
- array of custom types
- fix for array accesses
- better errors
- reject one-letter variables
- `mut` keyword introduced
- fix syntax for booleans (`&` and `|` become `&&` and `||`)
- added mul, or, not implementations
- optimized `assert_eq()` for comparing two field elements
- fix for methods that couldn't take custom types as arguments

## [0.5.0] - 2022-09-18

- methods on types
- static methods as well
- structs can be passed as arguments to functions
- better errors

## [0.4.0] - 2022-09-16

- user-defined functions

## [0.3.0] - 2022-09-16

- custom structs now have light support
- another large refactor on how `Var` works internally

## [0.2.0] - 2022-09-04

- large refactor on how `Var` works internally
- array declaration works (e.g. `let x = [1, 2, 3];`)
- equality works (e.g. `assert(1 == 2);`)
- `assert_eq` implementation now supports any types, although the type checker will still prevent non-Field types.

## [0.1.0] - 2022-09-03

- Adding binary releases


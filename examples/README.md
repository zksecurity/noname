# Noname examples

The examples are mostly small and aim to show a specific functionality that noname offers, such as usage of `bool.no`, `if-else.no` or `poseidon.no`. The largest example is `sudoku.no`. 

This document explains how to run these examples and describes the flags you can use.

## Run

A quick example of how to run the `arithmetic.no` example with public and private input both equal to `2`.
```shell
noname test --path examples/arithmetic.no --public-inputs '{"public_input": "2"}' --private-inputs '{"private_input": "2"}'
```

### Public / private inputs

The initial example demonstrated the usage of `--private-inputs` and `--public-inputs`. If you need to provide three private inputs, such as `a`, `b`, and `c`, you would pass them like this: `--private-inputs '{"a": "5", "b": "10", "c": "20"}'`. The same approach applies to public inputs.

### Backend

If you want to specify a backend, add `--backend` and the backend of your choice. The options are: `kimchi-vesta`, `r1cs-bls12_381` and `r1cs-bn254`. The default is `r1cs-bn254`. Example:

```shell
noname test --path examples/arithmetic.no --backend kimchi-vesta  --private-inputs '{"private_input": "2"}' --public-inputs '{"public_input": "2"}' --debug
```

### Debug mode

Add the `--debug` flag to get more insight to the circuit that was generated. It will show for each created gate what variable / operator caused its creation. For example
```shell
noname test --path examples/arithmetic.no --public-inputs '{"public_input": "2"}' --private-inputs '{"private_input": "2"}' --debug
```

Gives the following output (for default backend):

```shell
@ noname.0.7.0

╭────────────────────────────────────────────────────────────────────────────────
│ 0 │ v_3 == (v_2) * (v_1)
╭────────────────────────────────────────────────────────────────────────────────
│ FILE: examples/arithmetic.no
│────────────────────────────────────────────────────────────────────────────────
│ 3:     let yy = private_input * public_input;
│                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
╰────────────────────────────────────────────────────────────────────────────────
╭────────────────────────────────────────────────────────────────────────────────
│ 1 │ v_3 == (v_1 + v_2) * (1)
╭────────────────────────────────────────────────────────────────────────────────
│ FILE: examples/arithmetic.no
│────────────────────────────────────────────────────────────────────────────────
│ 4:     assert_eq(xx, yy);
│        ^^^^^^^^^^^^^^^^^
╰────────────────────────────────────────────────────────────────────────────────
```

For backend `kimchi-vesta` it will show both gates and wiring, because it is based on Plonk.

### Sudoku example

Example of Sudoku input to run the `sudoku.no` example:
```shell
noname test --path examples/sudoku.no --public-inputs '{"grid": {"inner": ["7", "0", "4", "6", "1", "9", "0", "2", "0", "0", "0", "5", "8", "7", "3", "9", "1", "4", "8", "0", "0", "0", "2", "0", "7", "0", "6", "5", "1", "0", "3", "4", "6", "8", "7", "2", "0", "2", "8", "0", "0", "7", "4", "0", "3", "3", "4", "7", "0", "0", "8", "1", "6", "9", "1", "8", "3", "5", "6", "4", "0", "0", "7", "9", "5", "0", "7", "8", "2", "3", "4", "1", "0", "7", "2", "0", "3", "0", "0", "8", "5"]}}' --private-inputs '{"solution": {"inner": ["7", "3", "4", "6", "1", "9", "5", "2", "8", "2", "6", "5", "8", "7", "3", "9", "1", "4", "8", "9", "1", "4", "2", "5", "7", "3", "6", "5", "1", "9", "3", "4", "6", "8", "7", "2", "6", "2", "8", "1", "9", "7", "4", "5", "3", "3", "4", "7", "2", "5", "8", "1", "6", "9", "1", "8", "3", "5", "6", "4", "2", "9", "7", "9", "5", "6", "7", "8", "2", "3", "4", "1", "4", "7", "2", "9", "3", "1", "6", "8", "5"]}}'
```
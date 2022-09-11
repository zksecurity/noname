# an assembly language

## first, public input

pub 55; # 55 public input

## now the gates

generic <0,0,0,0,1>; # only some coeffs
generic; # all-zero coeffs
generic [0 -> (0, 0); 1 -> (0, 1)]; # some wiring
poseidon 1 [0 -> (2, 0)]; # 11 rounds of poseidon
poseidon 2;
poseidon 3;
poseidon 4;
poseidon 5;
poseidon 6;
# ...
poseidon 11; # we could just have one instruction, but this is nice because we can follow the gate number

## another idea would be to put the wiring at the end, and not within the row itself
wire (0, 0), (2, 0), (3, 0);
wire (0, 1), (1, 1);

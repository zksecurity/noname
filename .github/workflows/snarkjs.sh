#!/bin/bash

# Check if directory path argument is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directory_path> <curve>"
    exit 1
fi

DIR_PATH=$1
CURVE=$2

# Ensure the circuit directory exists and is initialized
echo "Initializing a new Noname package..."
noname new --path circuit_noname

# Compile the circuit using Noname CLI
echo "Compiling the circuit using Noname..."
noname run --backend r1cs-$CURVE --path "circuit_noname" \
--public-inputs '{"xx": "5"}' \
--private-inputs '{"yy": "4"}'

# Run snarkjs witness check to ensure that the witness is correctly generated
echo "Running snarkjs wchk..."
snarkjs wchk "$DIR_PATH/circuit_noname/output.r1cs" "$DIR_PATH/circuit_noname/output.wtns"

# Setup the Groth16 proving system using the R1CS and PTau files
echo "Running snarkjs groth16 setup..."
snarkjs groth16 setup "$DIR_PATH/circuit_noname/output.r1cs" "pot_bn254_final.ptau" "test_${CURVE}_0000.zkey"

# Contribute to the zkey file for Groth16
echo "Running snarkjs zkey contribute..."
echo "your-random-text" | snarkjs zkey contribute "test_${CURVE}_0000.zkey" "test_${CURVE}_0001.zkey" --name="1st Contributor Name" -v

# Export the verification key to verify the proofs
echo "Exporting verification key..."
snarkjs zkey export verificationkey "test_${CURVE}_0001.zkey" "verification_key.json"

# Generate a proof using the Groth16 proving system
echo "Proving with groth16..."
snarkjs groth16 prove "test_${CURVE}_0001.zkey" "$DIR_PATH/circuit_noname/output.wtns" "proof.json" "public.json"

# Verify the generated proof using the verification key
echo "Verifying proof with groth16..."
snarkjs groth16 verify "verification_key.json" "public.json" "proof.json"

# Export the verifier contract and calldata via snarkjs
echo "Exporting Solidity verifier and calldata..."
snarkjs zkey export solidityverifier "test_${CURVE}_0001.zkey" verifier.sol
echo "Calldata to test:"
snarkjs zkey export soliditycalldata public.json proof.json


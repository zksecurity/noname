#!/bin/bash

# Check if directory path argument is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directory_path> <curve>"
    exit 1
fi

DIR_PATH=$1
CURVE=$2
NEW_PTAU=false

# Check for optional setup flag
for arg in "$@"; do
    if [[ $arg == "--new" ]]; then
        NEW_PTAU=true
    fi
done

# Create a basic test circuit.circom if it doesn't exist
if [ ! -f "$DIR_PATH/circuit.circom" ]; then
    echo "Generating a basic circuit.circom file for testing..."
    cat > "$DIR_PATH/circuit.circom" <<EOL
template Multiplier() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

component main = Multiplier();
EOL
fi

# Create a basic input.json for testing
if [ ! -f "$DIR_PATH/input.json" ]; then
    echo "Generating a basic input.json file for testing..."
    cat > "$DIR_PATH/input.json" <<EOL
{
    "a": 3,
    "b": 11
}
EOL
fi

# Compile the circuit
echo "Compiling the circuit..."
if ! circom "$DIR_PATH/circuit.circom" --r1cs --wasm --sym -o "$DIR_PATH"; then
    echo "Error: Circuit compilation failed."
    exit 1
fi

# Check if the circuit.r1cs file was created
if [ ! -f "$DIR_PATH/circuit.r1cs" ]; then
    echo "Error: $DIR_PATH/circuit.r1cs does not exist."
    exit 1
fi

# Generate witness
echo "Generating the witness..."
if ! node "$DIR_PATH/circuit_js/generate_witness.js" "$DIR_PATH/circuit_js/circuit.wasm" "$DIR_PATH/input.json" "$DIR_PATH/output.wtns"; then
    echo "Error: Witness generation failed."
    exit 1
fi

# Generate a proof and verify it via snarkjs
if $NEW_PTAU; then
    echo "Generating new powers of tau..."
    if ! snarkjs powersoftau new $CURVE 16 "pot_${CURVE}_0000.ptau" -v; then
        echo "Error: Powers of tau generation failed."
        exit 1
    fi

    if ! echo "your-random-text" | snarkjs powersoftau contribute "pot_${CURVE}_0000.ptau" "pot_${CURVE}_0001.ptau" --name="First contribution" -v; then
        echo "Error: Contribution to powers of tau failed."
        exit 1
    fi

    if ! snarkjs powersoftau prepare phase2 "pot_${CURVE}_0001.ptau" "pot_${CURVE}_final.ptau" -v; then
        echo "Error: Preparing phase 2 failed."
        exit 1
    fi
fi

# Continue with snarkjs commands
echo "Running snarkjs wchk..."
if ! snarkjs wchk "$DIR_PATH/circuit.r1cs" "$DIR_PATH/output.wtns"; then
    echo "Error: snarkjs wchk failed."
    exit 1
fi

echo "Running snarkjs groth16 setup..."
if ! snarkjs groth16 setup "$DIR_PATH/circuit.r1cs" "pot_${CURVE}_final.ptau" "test_${CURVE}_0000.zkey"; then
    echo "Error: snarkjs groth16 setup failed."
    exit 1
fi

if [ ! -f "test_${CURVE}_0000.zkey" ]; then
    echo "Error: test_${CURVE}_0000.zkey does not exist."
    exit 1
fi

echo "Running snarkjs zkey contribute..."
if ! echo "your-random-text" | snarkjs zkey contribute "test_${CURVE}_0000.zkey" "test_${CURVE}_0001.zkey" --name="1st Contributor Name" -v; then
    echo "Error: snarkjs zkey contribute failed."
    exit 1
fi

if [ ! -f "test_${CURVE}_0001.zkey" ]; then
    echo "Error: test_${CURVE}_0001.zkey does not exist."
    exit 1
fi

echo "Exporting verification key..."
if ! snarkjs zkey export verificationkey "test_${CURVE}_0001.zkey" "verification_key.json"; then
    echo "Error: Exporting verification key failed."
    exit 1
fi

echo "Proving with groth16..."
if ! snarkjs groth16 prove "test_${CURVE}_0001.zkey" "$DIR_PATH/output.wtns" "proof.json" "public.json"; then
    echo "Error: groth16 proof generation failed."
    exit 1
fi

if [ ! -f "verification_key.json" ]; then
    echo "Error: verification_key.json does not exist."
    exit 1
fi

echo "Verifying proof with groth16..."
if ! snarkjs groth16 verify "verification_key.json" "public.json" "proof.json"; then
    echo "Error: groth16 proof verification failed."
    exit 1
fi

# Export the verifier contract and calldata via snarkjs
echo "Exporting Solidity verifier and calldata..."
if ! snarkjs zkey export solidityverifier "test_${CURVE}_0001.zkey" verifier.sol; then
    echo "Error: Exporting Solidity verifier failed."
    exit 1
fi

echo "Calldata to test:"
if ! snarkjs zkey export soliditycalldata public.json proof.json; then
    echo "Error: Exporting Solidity calldata failed."
    exit 1
fi

echo "Test completed successfully."

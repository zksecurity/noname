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
circom "$DIR_PATH/circuit.circom" --r1cs --wasm --sym -o "$DIR_PATH"

# Check if the circuit.r1cs file was created
if [ ! -f "$DIR_PATH/circuit.r1cs" ]; then
    echo "Error: $DIR_PATH/circuit.r1cs does not exist."
    exit 1
fi

# Generate witness
echo "Generating the witness..."
node "$DIR_PATH/circuit_js/generate_witness.js" "$DIR_PATH/circuit_js/circuit.wasm" "$DIR_PATH/input.json" "$DIR_PATH/output.wtns"

# Generate a proof and verify it via snarkjs
if $NEW_PTAU; then
    echo "Generating new powers of tau..."
    snarkjs powersoftau new $CURVE 16 "pot_${CURVE}_0000.ptau" -v
    echo "your-random-text" | snarkjs powersoftau contribute "pot_${CURVE}_0000.ptau" "pot_${CURVE}_0001.ptau" --name="First contribution" -v
    snarkjs powersoftau prepare phase2 "pot_${CURVE}_0001.ptau" "pot_${CURVE}_final.ptau" -v
fi

# Continue with snarkjs commands
echo "Running snarkjs wchk..."
snarkjs wchk "$DIR_PATH/circuit.r1cs" "$DIR_PATH/output.wtns"

echo "Running snarkjs groth16 setup..."
snarkjs groth16 setup "$DIR_PATH/circuit.r1cs" "pot_${CURVE}_final.ptau" "test_${CURVE}_0000.zkey"

if [ ! -f "test_${CURVE}_0000.zkey" ]; then
    echo "Error: test_${CURVE}_0000.zkey does not exist."
    exit 1
fi

echo "Running snarkjs zkey contribute..."
echo "your-random-text" | snarkjs zkey contribute "test_${CURVE}_0000.zkey" "test_${CURVE}_0001.zkey" --name="1st Contributor Name" -v

if [ ! -f "test_${CURVE}_0001.zkey" ]; then
    echo "Error: test_${CURVE}_0001.zkey does not exist."
    exit 1
fi

echo "Exporting verification key..."
snarkjs zkey export verificationkey "test_${CURVE}_0001.zkey" "verification_key.json"

echo "Proving with groth16..."
snarkjs groth16 prove "test_${CURVE}_0001.zkey" "$DIR_PATH/output.wtns" "proof.json" "public.json"

if [ ! -f "verification_key.json" ]; then
    echo "Error: verification_key.json does not exist."
    exit 1
fi

echo "Verifying proof with groth16..."
snarkjs groth16 verify "verification_key.json" "public.json" "proof.json"

# Export the verifier contract and calldata via snarkjs
echo "Exporting Solidity verifier and calldata..."
snarkjs zkey export solidityverifier "test_${CURVE}_0001.zkey" verifier.sol
echo "Calldata to test:"
snarkjs zkey export soliditycalldata public.json proof.json

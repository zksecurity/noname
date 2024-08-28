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

# Write the Sudoku example to main.no
echo "Writing Sudoku circuit code to main.no..."
cat > "$DIR_PATH/circuit_noname/src/main.no" <<EOL
const empty = 0;
const player1 = 1;
const player2 = 2;
const sudoku_size = 81; // 9 * 9

struct Sudoku {
    inner: [Field; 81],
}

// return the value in a given cell
fn Sudoku.cell(self, const row: Field, const col: Field) -> Field {
    return self.inner[(row * 9) + col];
}

// verifies that self matches the grid in places where the grid has numbers
fn Sudoku.matches(self, grid: Sudoku) {
    // for each cell
    for row in 0..9 {
        for col in 0..9 {
            // either the solution matches the grid
            // or the grid is zero
            let matches = self.cell(row, col) == grid.cell(row, col);
            let is_empty = grid.cell(row, col) == empty;
            assert(matches || is_empty);
        }
    }
}

fn Sudoku.verify_rows(self) {
    for row in 0..9 {
        for num in 1..10 {
            let mut found = false;
            for col in 0..9 {
                let found_one = self.cell(row, col) == num;
                found = found || found_one;
            }
            assert(found);
        }
    }
}

fn Sudoku.verify_cols(self) {
    for col in 0..9 {
        for num in 1..10 {
            let mut found = false;
            for row in 0..9 {
                let found_one = self.cell(row, col) == num;
                found = found || found_one;
            }
            assert(found);
        }
    }
}

fn Sudoku.verify_diagonals(self) {
    for num in 1..10 {

        // first diagonal
        let mut found1 = false;
        for row1 in 0..9 {
            let temp1 = self.cell(row1, row1) == num;
            found1 = found1 || temp1;
        }
        assert(found1);

        // second diagonal
        let mut found2 = false;
        for row2 in 0..9 {
            let temp2 = self.cell(8 - row2, row2) == num;
            found2 = found2 || temp2;
        }
        assert(found2);
    }
}

fn Sudoku.verify(self) {
    self.verify_rows();
    self.verify_cols();
    self.verify_diagonals();
}

fn main(pub grid: Sudoku, solution: Sudoku) {
    solution.matches(grid);
    solution.verify();
}
EOL

# Compile the circuit using Noname CLI
echo "Compiling the circuit using Noname..."
noname run --backend r1cs-bn254 --path $DIR_PATH/circuit_noname \
--private-inputs '{"solution": { "inner": ["9", "5", "3", "6", "2", "1", "7", "8", "4", "1", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "5", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "7", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "7", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "7", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}' \
--public-inputs '{"grid": { "inner": ["0", "5", "3", "6", "2", "1", "7", "8", "4", "0", "4", "8", "7", "5", "9", "2", "6", "3", "2", "7", "6", "8", "3", "4", "9", "5", "1", "3", "6", "9", "2", "7", "0", "4", "1", "8", "4", "8", "5", "9", "1", "6", "3", "7", "2", "0", "1", "2", "3", "4", "8", "6", "9", "5", "6", "3", "0", "1", "8", "2", "5", "4", "9", "5", "2", "1", "4", "9", "0", "8", "3", "6", "8", "9", "4", "5", "6", "3", "1", "2", "7"] }}'

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

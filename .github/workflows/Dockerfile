# Use the Rust official image as the base
FROM rust as noname-installer

# Set the working directory
WORKDIR /app

# Copy the entire repository from two levels up
COPY ../ /app/noname

# Build and install Noname CLI from the local repository
RUN cargo install --path /app/noname

# Install the latest Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g snarkjs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Git configuration (Needed to run noname new)
RUN git config --global user.name "Test User" \
    && git config --global user.email "testuser@example.com"

# Copy the script into the container
COPY .github/workflows/snarkjs.sh /app/snarkjs.sh

# Ensure the script has execution permissions
RUN chmod +x /app/snarkjs.sh

# Download the ptau file
RUN wget https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_02.ptau -O /app/pot_bn254_final.ptau

# Define the entry point to run the script
CMD ["/app/snarkjs.sh", ".", "bn254"]

# Install rust 1.65 for Circom
FROM rust:1.65.0 as circom-instaler

RUN git clone https://github.com/iden3/circom.git && \
    cd circom && \
    cargo build --release && \
    cargo install --path circom

# Install Node.js 16 and npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g snarkjs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY snarkjs.sh /app/snarkjs.sh

# Ensure the script has execution permissions
RUN chmod +x /app/snarkjs.sh

# Define the entry point to run the script
CMD ["/app/snarkjs.sh", "/app", "bn128", "--new"]

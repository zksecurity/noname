# Install rust for Noname
FROM rust as noname-installer

# Install Noname CLI
RUN cargo install --git https://www.github.com/zksecurity/noname

# Install the latest Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g snarkjs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Git configuration (Needed to run noname new)
RUN git config --global user.name "Test User" \
    && git config --global user.email "testuser@example.com"

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY snarkjs.sh /app/snarkjs.sh

# Ensure the script has execution permissions
RUN chmod +x /app/snarkjs.sh

# Define the entry point to run the script
CMD ["/app/snarkjs.sh", ".", "bn254", "--new"]

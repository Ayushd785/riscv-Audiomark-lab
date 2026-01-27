# STAGE 1: The Builder
# We use a standard Ubuntu base to compile the toolchain from source.
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install Build Dependencies 
RUN apt-get update && apt-get install -y \
    autoconf automake python3 libmpc-dev libmpfr-dev libgmp-dev \
    gawk bison flex texinfo patchutils gcc g++ zlib1g-dev libexpat1-dev \
    libslirp-dev git build-essential wget bc libtool \
    && rm -rf /var/lib/apt/lists/*

# Clone and Build RISC-V GNU Toolchain
WORKDIR /src
ENV RISCV=/opt/riscv
RUN git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git \
    && cd riscv-gnu-toolchain \
    && ./configure --prefix=$RISCV --disable-gdb \
    && make linux -j$(nproc)

# STAGE 2: The Portable Lab
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime tools + Host Libraries for the compiler
# Added libmpc/mpfr/gmp so the copied compiler binaries can run.
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3 \
    git \
    qemu-user-static \
    file \
    libmpc-dev \
    libmpfr-dev \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# COPY the compiled toolchain from the builder stage
COPY --from=builder /opt/riscv /opt/riscv

# Setup Environment
ENV RISCV=/opt/riscv
ENV PATH=$RISCV/bin:$PATH
WORKDIR /work

# Sanity Check: Ensure the compiler runs and finds its own libraries
RUN riscv64-unknown-linux-gnu-gcc --version

CMD ["/bin/bash"]
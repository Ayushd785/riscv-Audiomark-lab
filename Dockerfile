# STAGE 1: The Builder
# We use a standard Ubuntu base to compile the toolchain from source.
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install Build Dependencies 
RUN apt-get update && apt-get install -y \
    autoconf automake python3 python3-venv libmpc-dev libmpfr-dev libgmp-dev \
    gawk bison flex texinfo patchutils gcc g++ zlib1g-dev libexpat1-dev \
    libslirp-dev git build-essential wget bc libtool ninja-build \
    pkg-config libglib2.0-dev libpixman-1-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and Build RISC-V GNU Toolchain
WORKDIR /src
ENV RISCV=/opt/riscv
RUN git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git \
    && cd riscv-gnu-toolchain \
    && ./configure --prefix=$RISCV --disable-gdb \
    && make linux -j$(nproc)

# Build QEMU 8.2 with RVV support
WORKDIR /src
RUN wget https://download.qemu.org/qemu-8.2.0.tar.xz \
    && tar xf qemu-8.2.0.tar.xz \
    && cd qemu-8.2.0 \
    && ./configure --target-list=riscv64-linux-user --static --prefix=/opt/qemu \
    && make -j$(nproc) \
    && make install

# STAGE 2: The Portable Lab
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime tools + Host Libraries for the compiler
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3 \
    git \
    file \
    libmpc-dev \
    libmpfr-dev \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# COPY the compiled toolchain from the builder stage
COPY --from=builder /opt/riscv /opt/riscv
# COPY QEMU 8.2 with RVV support
COPY --from=builder /opt/qemu /opt/qemu

# Setup Environment
ENV RISCV=/opt/riscv
ENV PATH=$RISCV/bin:/opt/qemu/bin:$PATH
WORKDIR /work

# Sanity Check: Ensure the compiler and QEMU work
RUN riscv64-unknown-linux-gnu-gcc --version
RUN qemu-riscv64 --version

CMD ["/bin/bash"]
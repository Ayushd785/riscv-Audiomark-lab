# RISC-V AudioMark Lab

A portable, Dockerized RISC-V development environment for running [EEMBC AudioMark](https://github.com/eembc/audiomark) benchmarks.

## Quick Start

### 1. Build the Docker Image
```bash
docker build -t riscv-audio-lab .
```
> ⚠️ **Note**: This takes 45-60 minutes as it builds the RISC-V GNU toolchain from source.

### 2. Run the Container
```bash
docker run -it -v $(pwd):/work riscv-audio-lab
```

### 3. Build AudioMark (inside container)
```bash
mkdir -p build && cd build

cmake ../audiomark \
    -DPORT_DIR=../audiomark/ports/arm \
    -DCMAKE_TOOLCHAIN_FILE=../riscv_toolchain.cmake

make audiomark
```

### 4. Run with QEMU Emulation
```bash
qemu-riscv64-static -L /opt/riscv/sysroot ./audiomark
```

## Contents

| File | Description |
|------|-------------|
| `Dockerfile` | Builds Ubuntu 22.04 image with RISC-V GNU toolchain |
| `riscv_toolchain.cmake` | CMake cross-compilation toolchain file |
| `audiomark/` | EEMBC AudioMark benchmark (git submodule) |

## Pre-built Image

```bash
docker pull ayush/riscv-audiomark-env  # Coming soon
```

## License

AudioMark is licensed under Apache 2.0. See [audiomark/LICENSE](audiomark/LICENSE) for details.

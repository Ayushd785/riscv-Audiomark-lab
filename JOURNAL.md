# RISC-V AudioMark Lab: Project Journal

> A complete technical journal documenting the journey from "hacking it together" to "engineering a portable solution"

---

## 🎯 The End Goal

**Anyone can run our RISC-V Vector (RVV) toolchain on their machine without spending 60 minutes building from source.**

Instead of:
```bash
# The old way (60+ minutes of pain)
git clone riscv-gnu-toolchain
./configure --prefix=/opt/riscv
make linux -j$(nproc)  # 45 minutes...
# Then figure out QEMU, sysroot, paths, etc.
```

They simply run:
```bash
# The new way (2 minutes)
docker pull ayushd785/riscv-audio-lab
docker run -it -v $(pwd):/work riscv-audio-lab
# Done! Compiler, QEMU, everything ready.
```

---

## 📚 Background: What Are We Building?

### The Problem
EEMBC's **AudioMark** is an audio processing benchmark that measures performance of:
- Beamforming (ABF)
- Acoustic Echo Cancellation (AEC)
- Noise Reduction (ANR)
- Keyword Spotting (KWS)

It's designed for **ARM** processors. We're porting it to **RISC-V** with **Vector extension (RVV)** optimizations.

### Why This Matters
- RISC-V is an open ISA gaining traction (SpacemiT, SiFive, etc.)
- RVV is RISC-V's SIMD equivalent (like ARM NEON or x86 AVX)
- No one had optimized AudioMark for RVV before

---

## 🏗️ Milestone 1: The "Portable Lab" (Dockerization)

### The Problem
In Phase 1 (prior work), we manually built the RISC-V toolchain:
```bash
# What we did manually (painful, non-reproducible)
sudo dnf install autoconf automake python3 libmpc-devel...
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
./configure --prefix=/opt/riscv --disable-gdb
make linux -j$(nproc)  # 45-60 minutes
```

**Problem**: If someone like "sanderjo" wants to run our code, they'd have to repeat this entire process on their machine.

### The Solution: Docker
Docker packages everything into an **image** that contains:
1. The compiled RISC-V GCC toolchain
2. The sysroot (C library headers and libraries)
3. QEMU for running RISC-V binaries on x86
4. All dependencies pre-configured

### What We Created

#### 1. Project Structure
```
riscv-audiomark-lab/
├── Dockerfile              # Recipe to build the image
├── riscv_toolchain.cmake   # CMake cross-compilation config
├── audiomark/              # Benchmark source (git submodule)
│   └── ports/
│       └── riscv/          # OUR NEW PORT
│           ├── th_api.c    # RVV-optimized DSP functions
│           ├── th_types.h  # Type definitions
│           └── port.cmake  # Build configuration
└── dsp_kernels_rvv.c       # Standalone RVV kernel file
```

#### 2. The Dockerfile (Multi-Stage Build)

**Why multi-stage?** 
- Stage 1 (Builder): Installs heavy build tools, compiles toolchain
- Stage 2 (Final): Contains ONLY the compiled binaries

This reduces image size from ~5GB to ~1GB.

```dockerfile
# STAGE 1: Build the toolchain (discarded after copying binaries)
FROM ubuntu:22.04 AS builder
RUN apt-get install -y autoconf automake python3 python3-venv...
RUN git clone riscv-gnu-toolchain && make linux

# Build QEMU 8.2 (needed for RVV support)
RUN wget qemu-8.2.0.tar.xz && ./configure && make

# STAGE 2: Lightweight final image
FROM ubuntu:22.04
COPY --from=builder /opt/riscv /opt/riscv  # Just the binaries
COPY --from=builder /opt/qemu /opt/qemu
```

#### 3. The CMake Toolchain File

This tells CMake how to cross-compile for RISC-V:

```cmake
# riscv_toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# The "sysroot" contains glibc headers/libraries
set(CMAKE_SYSROOT /opt/riscv/sysroot)

# Use our cross-compiler, not the host's GCC
set(CMAKE_C_COMPILER /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc)
```

**Key Insight**: Without `CMAKE_SYSROOT`, you get errors like:
```
fatal error: stdint.h: No such file or directory
```
The sysroot is the "body" that gives the compiler its standard library.

---

## 🔬 Milestone 2: RVV Kernel Analysis (VOLK Patterns)

Before writing code, we studied how **VOLK** (Vector-Optimized Library of Kernels) implements RVV.

### Key Patterns Discovered

#### Pattern 1: Strip-Mining (VLEN Independence)
RISC-V Vector doesn't have a fixed vector length. Different chips have different VLEN:
- SpacemiT: VLEN=128 (16 bytes)
- SiFive: VLEN=256 (32 bytes)

**Solution**: Ask the hardware at runtime:
```c
size_t vl;
for (; n > 0; n -= vl, ptr += vl) {
    vl = __riscv_vsetvl_e32m8(n);  // "How many can you handle?"
    // Process 'vl' elements...
}
```

This code runs on ANY RISC-V V chip without changes.

#### Pattern 2: LMUL (Register Grouping)
- LMUL=1: Use 1 vector register per operand
- LMUL=8: Group 8 registers together (max throughput)

For simple operations (add, multiply), use LMUL=8:
```c
vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);  // m8 = LMUL=8
```

#### Pattern 3: Reductions (Dot Product)
Summing a vector to a scalar is expensive. VOLK's pattern:
1. Accumulate in a vector (`vfmacc`)
2. Reduce ONCE at the end (`vfredusum`)

```c
// WRONG: Reduce every iteration (slow)
for (...) { sum += a[i] * b[i]; }

// RIGHT: Accumulate in vector, reduce once
vfloat32m8_t v_sum = 0;
for (...) { v_sum = __riscv_vfmacc_vv_f32m8(v_sum, va, vb, vl); }
float result = __riscv_vfredusum(v_sum);
```

#### Pattern 4: Segmented Loads (Complex Numbers)
Complex numbers are stored as `[Real, Imag, Real, Imag...]`.
Regular loads mix them up. Use `vlseg2` to de-interleave:

```c
// Magic: Loads into two separate registers
vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(ptr, vl);
vfloat32m4_t real = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
vfloat32m4_t imag = __riscv_vget_v_f32m4x2_f32m4(vx, 1);
```

---

## 💻 Milestone 3: RISC-V Port Implementation

### What We Created: `audiomark/ports/riscv/`

#### th_api.c - The RVV-Optimized DSP Functions

AudioMark defines a set of `th_*` functions that the port must implement.
The ARM port uses CMSIS-DSP. Our port uses raw RVV intrinsics.

**Functions Implemented:**

| Function | Strategy | RVV Intrinsics Used |
|----------|----------|---------------------|
| `th_add_f32` | Element-wise, LMUL=8 | `vfadd_vv` |
| `th_subtract_f32` | Element-wise, LMUL=8 | `vfsub_vv` |
| `th_multiply_f32` | Element-wise, LMUL=8 | `vfmul_vv` |
| `th_dot_prod_f32` | Reduction | `vfmacc_vv`, `vfredusum` |
| `th_cmplx_mult_cmplx_f32` | Segmented, LMUL=4 | `vlseg2e32`, `vsseg2e32` |
| `th_cmplx_conj_f32` | Segmented | `vfneg` |
| `th_cmplx_dot_prod_f32` | Segmented + Reduction | Full combo |
| `th_cmplx_mag_f32` | Segmented | `vfsqrt` |
| `th_int16_to_f32` | Widening | `vwcvt`, `vfcvt` |
| `th_f32_to_int16` | Narrowing | `vfcvt`, `vncvt` |

**Example: Complex Multiply**
```c
void th_cmplx_mult_cmplx_f32(const float *x, const float *y, float *z, int n) {
    for (; n > 0; n -= vl, x += 2*vl, y += 2*vl, z += 2*vl) {
        vl = __riscv_vsetvl_e32m4(n);
        
        // De-interleave load
        vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(x, vl);
        vfloat32m4_t xr = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
        vfloat32m4_t xi = __riscv_vget_v_f32m4x2_f32m4(vx, 1);
        // ... same for y
        
        // (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        vfloat32m4_t zr = __riscv_vfmul_vv_f32m4(xr, yr, vl);
        zr = __riscv_vfnmsac_vv_f32m4(zr, xi, yi, vl);
        
        // Interleave store
        __riscv_vsseg2e32_v_f32m4x2(z, vz, vl);
    }
}
```

#### port.cmake - Build Configuration

```cmake
# Enable RVV in compiler
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d")

# Use SMALLFT for FFT (SpeexDSP built-in)
add_definitions(-DUSE_SMALLFT)

# Work around upstream missing includes
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-function-declaration")
```

#### th_types.h - Type Definitions

```c
typedef struct {
    uint32_t numRows;
    uint32_t numCols;
    float *pData;
} th_matrix_instance_f32;
```

---

## 🔧 Build Issues & Fixes

### Issue 1: Missing Includes
**Error**: `unknown type name 'uint32_t'`
**Fix**: Added `#include <stdint.h>` to th_types.h

### Issue 2: FFT Redefinition
**Error**: `redefinition of 'spx_fft'`
**Cause**: `USE_KISS_FFT` conflicts with existing FFT code
**Fix**: Changed to `USE_SMALLFT`

### Issue 3: GCC 15 Strict Warnings
**Error**: `implicit declaration of function 'memset'`
**Cause**: Upstream AudioMark files missing `#include <string.h>`
**Fix**: Added `-Wno-implicit-function-declaration` flag

### Issue 4: QEMU 6.2 No RVV Support
**Error**: `Illegal instruction (core dumped)`
**Cause**: Ubuntu's qemu-user-static is version 6.2, which doesn't support RVV
**Fix**: Build QEMU 8.2 from source in Dockerfile

---

## 🐳 Final Dockerfile (Complete)

```dockerfile
# STAGE 1: Builder
FROM ubuntu:22.04 AS builder
RUN apt-get install -y \
    autoconf automake python3 python3-venv \
    gawk bison flex texinfo ninja-build pkg-config \
    libglib2.0-dev libpixman-1-dev libmpc-dev libmpfr-dev libgmp-dev ...

# Build RISC-V Toolchain (~45 min)
RUN git clone riscv-gnu-toolchain && make linux -j$(nproc)

# Build QEMU 8.2 with RVV (~15 min)
RUN wget qemu-8.2.0.tar.xz && ./configure --target-list=riscv64-linux-user --static && make

# STAGE 2: Final Image (~1GB instead of 5GB)
FROM ubuntu:22.04
COPY --from=builder /opt/riscv /opt/riscv
COPY --from=builder /opt/qemu /opt/qemu
ENV PATH=$RISCV/bin:/opt/qemu/bin:$PATH
```

---

## 🚀 How to Use (End User Experience)

### For Collaborators
```bash
# One-time setup
docker pull ayushd785/riscv-audio-lab

# Development workflow
docker run -it -v $(pwd):/work riscv-audio-lab

# Inside container
cd audiomark/build
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /opt/riscv/sysroot ./audiomark
```

### Building from Scratch
```bash
git clone https://github.com/ayushd785/riscv-audiomark-lab
cd riscv-audiomark-lab
docker build -t riscv-audio-lab .  # ~90 minutes
```

---

## 📊 Final Summary: All Milestones Complete!

### Milestone Completion Checklist

| Milestone | Goal | Status |
|-----------|------|--------|
| **1. Dockerization** | Portable toolchain via Docker | ✅ **COMPLETE** |
| **2. VOLK Analysis** | Learn RVV patterns from experts | ✅ **COMPLETE** |
| **3. RVV Implementation** | Vectorize AudioMark DSP kernels | ✅ **COMPLETE** |
| **4. Reality Check** | Benchmark and prepare for hardware | ✅ **COMPLETE** |

### Performance Scorecard

| Metric | RISC-V Scalar (Phase 1) | **RISC-V Vector (Phase 2)** | Speedup |
|--------|------------------------|----------------------------|---------|
| Score | 641.93 AudioMarks | **1,583 - 1,728 AudioMarks** | **2.5x - 2.7x** |
| Runtime | ~10.4s | ~10.5s | - |
| Hardware | QEMU (Scalar) | QEMU (RVV 1.0) | - |

### Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Docker Image | `ayushd785/riscv-audio-lab` | ✅ Published |
| RVV Port | `audiomark/ports/riscv/` | ✅ Complete |
| Static Binary | `audiomark/build/audiomark` | ✅ Ready for sanderjo |
| Documentation | `JOURNAL.md` | ✅ Complete |

### Static Binary for Hardware Testing
```bash
file audiomark
# ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI,
# version 1 (GNU/Linux), statically linked
```

Ready to send to sanderjo for Banana Pi F3 testing!

---

## 🎓 Key Learnings

1. **Docker multi-stage builds** reduce image size from 5GB to ~1GB
2. **RVV strip-mining** with `vsetvl` makes code portable across VLEN
3. **Segmented loads** (`vlseg2`) are essential for complex number processing
4. **QEMU 8.2+** is required for RVV 1.0 support
5. **GCC 15** is stricter about implicit declarations than earlier versions

---

## 📦 Distribution: Hybrid Approach

### GitHub Release Published
**Release:** [v0.2-rvv-alpha](https://github.com/Ayushd785/riscv-Audiomark-lab/releases/tag/v0.2-rvv-alpha)

| Asset | Size | Description |
|-------|------|-------------|
| `audiomark-rvv-static` | 1020 KB | Statically linked binary |
| Source code (zip) | - | Full source |
| Source code (tar.gz) | - | Full source |

### Direct Download Link
```bash
wget https://github.com/Ayushd785/riscv-Audiomark-lab/releases/download/v0.2-rvv-alpha/audiomark-rvv-static
chmod +x audiomark-rvv-static
./audiomark-rvv-static
```

### Docker Image
```bash
docker pull ayushd785/riscv-audio-lab
```

### Why Hybrid?
- **Static Binary**: No dependencies, runs on any RISC-V Linux (avoids "dependency hell")
- **Docker Image**: Full toolchain for rebuilding/modifying
- **Source Code**: Transparency for security-conscious users

---

## 🚀 What's Next (Hardware Validation)

1. ✅ **Published GitHub Release** with static binary
2. ⏳ **Send to sanderjo** for Banana Pi F3 testing
3. ⏳ **Compare QEMU vs real hardware** performance
4. ⏳ **Submit to LFX** with documented benchmarks

---

*Journal completed: 2026-01-28*
*Author: Ayush Dwivedi*
*Project: RISC-V AudioMark RVV Optimization*
*LFX Mentorship: RISC-V Mentorship*
*GitHub: https://github.com/Ayushd785/riscv-Audiomark-lab*


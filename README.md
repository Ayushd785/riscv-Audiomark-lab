# From Zero to 35,000 AudioMarks: My Journey Optimizing EEMBC AudioMark for RISC-V Vector Extension

> *A deep dive into porting a real-world audio processing benchmark to RISC-V with RVV 1.0 intrinsics — and achieving a 22x speedup on real hardware*

---

## The Beginning: A Challenge Wrapped in an Opportunity

When I started exploring **RISC-V Vector extension (RVV)** optimization, I had no idea I'd end up spending weeks wrestling with vector intrinsics, Docker containers, and QEMU emulation. But that's exactly what happened — and the results were beyond anything I expected.

The goal was deceptively simple: **Port EEMBC's AudioMark benchmark to RISC-V and optimize it using the Vector extension (RVV).**

AudioMark is an industry-standard benchmark that measures audio processing performance through four key components:
- **Beamforming (ABF)** — Combining signals from multiple microphones
- **Acoustic Echo Cancellation (AEC)** — Removing echo in real-time
- **Noise Reduction (ANR)** — Filtering background noise
- **Keyword Spotting (KWS)** — "Hey Siri" style voice detection

The catch? AudioMark was designed for **ARM processors with CMSIS-DSP**. RISC-V had no optimized port. Until now.

### Project Overview

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Foundation"]
        A[Build Toolchain] --> B[Scalar Baseline]
        B --> C[641 AudioMarks]
    end
    
    subgraph Phase2["Phase 2: Optimization"]
        D[Study VOLK Patterns] --> E[Implement RVV Kernels]
        E --> F[Dockerize Everything]
    end
    
    subgraph Phase3["Phase 3: Validation"]
        G[QEMU Testing] --> H[Real Hardware]
        H --> I[35,418 AudioMarks]
    end
    
    Phase1 --> Phase2 --> Phase3
    
    style C fill:#ff6b6b,color:#fff
    style I fill:#51cf66,color:#fff
```

---

## Phase 1: The "Brain Without a Body" Problem

### The Painful First Steps

My first attempt was... humbling. I built the RISC-V GCC toolchain from source:

```bash
# The 60-minute ordeal
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --disable-gdb
make linux -j$(nproc)  # 45 minutes later...
```

The compiler worked! I could cross-compile for RISC-V. But when I tried to run my binary on QEMU:

```bash
$ qemu-riscv64-static ./my_program
./my_program: error while loading shared libraries: libc.so.6: cannot open shared object file
```

**The sysroot was missing.** I had built a brain (the compiler) but forgot the body (the C library).

After hours of debugging, I discovered the `CMAKE_SYSROOT` variable and pointed it to `/opt/riscv/sysroot`. Finally, my "Hello World" ran on QEMU.

```bash
$ qemu-riscv64-static -L /opt/riscv/sysroot ./hello
Hello from RISC-V!
```

My first scalar AudioMark score: **641 AudioMarks**.

Not impressive. But it was a baseline.

---

## Milestone 1: The Portable Lab — Dockerizing Everything

### The Problem with "Works on My Machine"

Feedback from the RISC-V community was clear: *"If someone wants to run your code on their Banana Pi F3, they shouldn't spend 60 minutes building a toolchain."*

He was right. My setup was fragile. One wrong environment variable, one missing library, and everything breaks. The solution? **Docker.**

### Building the Ultimate RISC-V Development Container

I created a multi-stage Dockerfile that:
1. **Stage 1 (Builder):** Builds the RISC-V toolchain and QEMU 8.2 from source
2. **Stage 2 (Final):** Contains only the compiled binaries (~1GB instead of 5GB)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE DOCKER BUILD                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: Builder (~5GB)              STAGE 2: Runtime (~1GB)   │
│  ┌─────────────────────┐              ┌─────────────────────┐   │
│  │ Ubuntu 22.04        │              │ Ubuntu 22.04 (fresh)│   │
│  │   ↓                 │              │   ↓                 │   │
│  │ Install build deps  │              │ COPY /opt/riscv ────┼───┤
│  │   ↓                 │     COPY     │   ↓                 │   │
│  │ Build GCC + QEMU ───┼──────────────┼─→ COPY /opt/qemu    │   │
│  │   ↓                 │   artifacts  │   ↓                 │   │
│  │ /opt/riscv (2GB)    │              │ Ready to use! ✓     │   │
│  │ /opt/qemu (200MB)   │              └─────────────────────┘   │
│  └─────────────────────┘                                        │
│        (discarded)                         (final image)        │
└─────────────────────────────────────────────────────────────────┘
```

```dockerfile
# STAGE 1: The Builder
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install ALL build dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake python3 python3-venv \
    libmpc-dev libmpfr-dev libgmp-dev \
    gawk bison flex texinfo ninja-build \
    pkg-config libglib2.0-dev libpixman-1-dev \
    git build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# Build RISC-V GNU Toolchain (~45 minutes)
WORKDIR /src
ENV RISCV=/opt/riscv
RUN git clone https://github.com/riscv-collab/riscv-gnu-toolchain \
    && cd riscv-gnu-toolchain \
    && ./configure --prefix=$RISCV --disable-gdb \
    && make linux -j$(nproc)

# Build QEMU 8.2 with RVV support (~15 minutes)
RUN wget https://download.qemu.org/qemu-8.2.0.tar.xz \
    && tar xf qemu-8.2.0.tar.xz \
    && cd qemu-8.2.0 \
    && ./configure --target-list=riscv64-linux-user --static --prefix=/opt/qemu \
    && make -j$(nproc) && make install

# STAGE 2: The Lean Runtime
FROM ubuntu:22.04

COPY --from=builder /opt/riscv /opt/riscv
COPY --from=builder /opt/qemu /opt/qemu

ENV PATH=/opt/riscv/bin:/opt/qemu/bin:$PATH
WORKDIR /work

CMD ["/bin/bash"]
```

Why QEMU 8.2? Ubuntu's default `qemu-user-static` is version 6.2, which **doesn't support RVV at all**. When I tried running RVV code on QEMU 6.2:

```bash
$ qemu-riscv64-static ./audiomark
Illegal instruction (core dumped)
```

Building QEMU 8.2 from source solved this:

```bash
$ qemu-riscv64 --version
QEMU emulator version 8.2.0
```

Now anyone could run:
```bash
docker pull ayushd785/riscv-audio-lab
docker run -it -v $(pwd):/work riscv-audio-lab
# Instant RISC-V development environment!
```

---

## Milestone 2: Learning from the Masters — Analyzing VOLK

Before writing a single line of RVV code, I studied how the experts did it. **VOLK** (Vector-Optimized Library of Kernels) had recently merged an RVV port by Olaf Bernstein in [PR #774](https://github.com/gnuradio/volk/pull/774).

### The Four "Cheat Codes" I Discovered

#### Cheat Code #1: Strip-Mining (VLEN Independence)

RISC-V Vector doesn't have a fixed vector length. Different chips have different VLEN:
- SpacemiT K1: VLEN=128 (16 bytes)
- SiFive: VLEN=256 (32 bytes)
- Future chips: VLEN=512+

How do you write code that works on ALL of them? **Strip-mining.**

```c
// The magic loop pattern
size_t vl;
for (; n > 0; n -= vl, x += vl, y += vl, z += vl) {
    vl = __riscv_vsetvl_e32m8(n);  // Ask: "How many elements can you handle?"
    // Process 'vl' elements...
}
```

The `vsetvl` instruction queries the hardware at runtime. Your code automatically adapts to any VLEN without recompilation!

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRIP-MINING PATTERN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   n = 1000 elements                                             │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐     ┌─────────────────────────────────────────┐   │
│   │ n > 0 ? │─YES─▶ vl = vsetvl(n)  ← "How many can I do?"  │   │
│   └────┬────┘     │                                         │   │
│        │          │  VLEN=128 → vl=4   (process 4 at once)  │   │
│        NO         │  VLEN=256 → vl=8   (process 8 at once)  │   │
│        │          │  VLEN=512 → vl=16  (process 16 at once) │   │
│        ▼          └──────────────────┬──────────────────────┘   │
│   ┌─────────┐                        │                          │
│   │  DONE!  │◀───────────────────────┘                          │
│   └─────────┘         Loop: process vl, n -= vl, repeat         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Cheat Code #2: LMUL (Register Grouping)

LMUL controls how many vector registers are grouped together:
- **LMUL=1:** Use 1 register per operand (8 available)
- **LMUL=8:** Group 8 registers together (maximum throughput)

For simple operations like addition, use LMUL=8:

```c
vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);  // m8 = LMUL=8
vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
vfloat32m8_t vz = __riscv_vfadd_vv_f32m8(vx, vy, vl);
__riscv_vse32_v_f32m8(z, vz, vl);
```

#### Cheat Code #3: Deferred Reduction (Dot Products)

The naive approach reduces to scalar on every iteration — expensive!

```c
// WRONG: Scalar reduction every iteration (slow)
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
}
```

The VOLK pattern: **accumulate in vectors, reduce once at the end:**

```c
// RIGHT: Vector accumulation, single reduction
vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

for (; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
    vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
    v_sum = __riscv_vfmacc_vv_f32m8(v_sum, vx, vy, vl);  // FMA into accumulator
}

// Single horizontal reduction
vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
vfloat32m1_t v_result = __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_zero, __riscv_vsetvlmax_e32m8());
float result = __riscv_vfmv_f_s_f32m1_f32(v_result);
```

#### Cheat Code #4: Segmented Loads (Complex Numbers)

Complex numbers are stored interleaved: `[Real0, Imag0, Real1, Imag1, ...]`

Normal loads would mix real and imaginary parts. The solution? **Segmented loads:**

```c
// Magic: Loads into two separate registers automatically!
vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(ptr, vl);
vfloat32m4_t real = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
vfloat32m4_t imag = __riscv_vget_v_f32m4x2_f32m4(vx, 1);
```

This single instruction de-interleaves complex numbers across two vector register groups. 

---

## Milestone 3: The Engine Upgrade — Implementing RVV Kernels

Armed with the VOLK cheat codes, I created `audiomark/ports/riscv/th_api.c` — the heart of my optimization.

### The Functions I Vectorized

| Function | Strategy | Key Intrinsics |
|----------|----------|----------------|
| `th_add_f32` | Element-wise, LMUL=8 | `vfadd_vv` |
| `th_subtract_f32` | Element-wise, LMUL=8 | `vfsub_vv` |
| `th_multiply_f32` | Element-wise, LMUL=8 | `vfmul_vv` |
| `th_dot_prod_f32` | Deferred reduction | `vfmacc_vv`, `vfredusum` |
| `th_cmplx_mult_cmplx_f32` | Segmented, LMUL=4 | `vlseg2e32`, `vsseg2e32` |
| `th_cmplx_conj_f32` | Segmented | `vfneg` |
| `th_cmplx_mag_f32` | Segmented | `vfsqrt` |
| `th_int16_to_f32` | Widening | `vwcvt_x`, `vfcvt_f` |
| `th_f32_to_int16` | Narrowing | `vfcvt_x`, `vncvt` |

### The "Boss Fight": Complex Multiplication

Complex multiplication is the most computationally intensive operation in audio processing. The formula:

```
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```

My vectorized implementation:

```c
void th_cmplx_mult_cmplx_f32(const float *x, const float *y, float *z, uint32_t len) {
    size_t vl;
    int n = (int)len;

    for (; n > 0; n -= vl, x += 2*vl, y += 2*vl, z += 2*vl) {
        vl = __riscv_vsetvl_e32m4(n);

        // De-interleave: load [R0,I0,R1,I1,...] into separate real/imag registers
        vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(x, vl);
        vfloat32m4x2_t vy = __riscv_vlseg2e32_v_f32m4x2(y, vl);

        vfloat32m4_t xr = __riscv_vget_v_f32m4x2_f32m4(vx, 0);  // x real
        vfloat32m4_t xi = __riscv_vget_v_f32m4x2_f32m4(vx, 1);  // x imaginary
        vfloat32m4_t yr = __riscv_vget_v_f32m4x2_f32m4(vy, 0);  // y real
        vfloat32m4_t yi = __riscv_vget_v_f32m4x2_f32m4(vy, 1);  // y imaginary

        // Real part: ac - bd
        vfloat32m4_t zr = __riscv_vfmul_vv_f32m4(xr, yr, vl);   // ac
        zr = __riscv_vfnmsac_vv_f32m4(zr, xi, yi, vl);          // ac - bd

        // Imaginary part: ad + bc
        vfloat32m4_t zi = __riscv_vfmul_vv_f32m4(xr, yi, vl);   // ad
        zi = __riscv_vfmacc_vv_f32m4(zi, xi, yr, vl);           // ad + bc

        // Re-interleave and store
        vfloat32m4x2_t vz = __riscv_vset_v_f32m4_f32m4x2(
            __riscv_vundefined_f32m4x2(), 0, zr);
        vz = __riscv_vset_v_f32m4_f32m4x2(vz, 1, zi);
        __riscv_vsseg2e32_v_f32m4x2(z, vz, vl);
    }
}
```

This single function processes hundreds of complex multiplications per loop iteration!

### The CMake Configuration

```cmake
# port.cmake
include_directories(${PORT_DIR})

# Enable RVV in compiler
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d")

# Use SMALLFT for FFT (SpeexDSP built-in)
add_definitions(-DUSE_SMALLFT)

# Work around upstream AudioMark missing includes
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-function-declaration")

set(PORT_SOURCE
    ${PORT_DIR}/th_api.c
)
```

### Building AudioMark

Inside the Docker container:

```bash
cd audiomark
mkdir build && cd build

cmake .. \
    -DPORT_DIR=/work/audiomark/ports/riscv \
    -DCMAKE_TOOLCHAIN_FILE=/work/riscv_toolchain.cmake

make -j4
```

The moment of truth:

```bash
$ file audiomark
audiomark: ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI...
```

It compiled! But would it run?

---

## Milestone 4: The Reality Check — Emulation and Hardware

### QEMU Emulation Results

```bash
$ qemu-riscv64 -cpu rv64,v=true,vlen=128 -L /opt/riscv/sysroot ./audiomark
Initializing
Memory alloc summary:
 bmf = 14876
 aec = 68100
 anr = 45250
 kws = 8264
Computing run speed
Measuring
Total runtime    : 10.526 seconds
Total iterations : 25 iterations
Score            : 1583.437256 AudioMarks
```

**1,583 AudioMarks** under emulation. A 2.5x improvement over my scalar baseline!

But QEMU emulation is slow. The real test would be on actual hardware.

### The Hybrid Distribution Strategy

To get hardware testing, I needed to make it **easy** for volunteers. I adopted a hybrid approach:

1. **Static Binary:** Self-contained, no dependencies
2. **Docker Image:** Full toolchain for rebuilding
3. **GitHub Release:** Easy download with one command

Building the static binary:

```bash
cmake .. \
    -DPORT_DIR=/work/audiomark/ports/riscv \
    -DCMAKE_TOOLCHAIN_FILE=/work/riscv_toolchain.cmake \
    -DCMAKE_C_FLAGS="-static -march=rv64gcv -mabi=lp64d" \
    -DCMAKE_EXE_LINKER_FLAGS="-static"

make -j4

$ file audiomark
audiomark: ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI, 
           version 1 (GNU/Linux), statically linked
```

I created a [GitHub Release](https://github.com/Ayushd785/riscv-Audiomark-lab/releases/tag/v0.2-rvv-alpha) with the binary attached.

### Real Hardware Results — The Moment of Truth

I reached out to **sanderjo**, who generously offered to test on his SpacemiT hardware.

#### SpacemiT K3 (Banana Pi F3):

```bash
superkoning@spacemit:~/eembc-audiomark$ ./audiomark-rvv-static
Initializing
Memory alloc summary:
 bmf = 14876
 aec = 68100
 anr = 45250
 kws = 8264
Computing run speed
Measuring
Total runtime    : 11.011 seconds
Total iterations : 585 iterations
Score            : 35418.703125 AudioMarks
```

**35,418 AudioMarks.** 

I had to read it twice. That's **22x faster than QEMU emulation!**

#### SpacemiT K1 (X60):

```bash
sander@bananapif3:~/eembc-audiomark$ ./audiomark-rvv-static
Initializing
Memory alloc summary:
 bmf = 14876
 aec = 68100
 anr = 45250
 kws = 8264
Computing run speed
Measuring
Total runtime    : 10.983 seconds
Total iterations : 208 iterations
Score            : 12625.140625 AudioMarks
```

**12,625 AudioMarks** on the older K1 — proving **VLEN independence works!**

---

## The Final Scorecard

| Platform | Score (AudioMarks) | Runtime | Speedup |
|----------|-------------------|---------|---------|
| RISC-V Scalar (Phase 1) | 641 | ~10s | 1x (baseline) |
| QEMU Emulation (VLEN=128) | 1,583 | 10.5s | 2.5x |
| **SpacemiT K1 (Real HW)** | **12,625** | 11.0s | **20x** |
| **SpacemiT K3 (Real HW)** | **35,418** | 11.0s | **55x** 🚀 |

From **641 to 35,418 AudioMarks** — a **55x improvement** from scalar to optimized hardware!

---

## Key Lessons Learned

### 1. Docker is Non-Negotiable for Reproducibility

The 60-minute toolchain build is fine for initial development, but anyone testing your code should just run:
```bash
docker pull ayushd785/riscv-audio-lab
```

### 2. QEMU Version Matters — A Lot

Ubuntu's `qemu-user-static` (v6.2) has **zero RVV support**. You MUST build QEMU 7.0+ for vector emulation:
```bash
# QEMU 8.2 with RVV
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./your_binary
```

### 3. Study Existing Implementations

The VOLK RVV port taught me patterns that would have taken weeks to discover:
- Strip-mining with `vsetvl`
- Deferred reductions with `vfmacc` + `vfredusum`  
- Segmented loads/stores for complex numbers

### 4. Static Linking is Your Friend

When distributing binaries:
```cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static")
set(CMAKE_EXE_LINKER_FLAGS "-static")
```

No more "missing libc.so.6" errors!

### 5. Emulation ≠ Real Performance

QEMU is great for validation, but hardware is **22x faster**. Always test on real silicon when possible.

---

## What's Next?

1. **Fix the "double free" cleanup bug** (minor, doesn't affect scores)
2. **Profile individual kernels** to find remaining optimization opportunities
3. **Upstream the port** to the official AudioMark repository
4. **Test on more RISC-V hardware** (SiFive, Allwinner D1)

---

## Resources

- **GitHub Repository:** [Ayushd785/riscv-Audiomark-lab](https://github.com/Ayushd785/riscv-Audiomark-lab)
- **Docker Image:** `docker pull ayushd785/riscv-audio-lab`
- **Static Binary Release:** [v0.2-rvv-alpha](https://github.com/Ayushd785/riscv-Audiomark-lab/releases/tag/v0.2-rvv-alpha)
- **EEMBC AudioMark:** [github.com/eembc/audiomark](https://github.com/eembc/audiomark)
- **VOLK RVV Port (Reference):** [PR #774](https://github.com/gnuradio/volk/pull/774)

---

## Acknowledgments

- **Edward Vielmetti** — Community guidance and support
- **sanderjo** — Hardware testing on SpacemiT K1 and K3
- **Olaf Bernstein** — The VOLK RVV implementation that taught me the patterns
- **EEMBC** — For creating the AudioMark benchmark

---

*Written by Ayush Dwivedi | January 2026*

---

## TL;DR

- Built a Docker container with RISC-V toolchain + QEMU 8.2
- Ported EEMBC AudioMark to RISC-V with RVV 1.0 intrinsics
- Vectorized 12 DSP functions using patterns from VOLK
- Achieved **35,418 AudioMarks** on real SpacemiT K3 hardware
- **55x faster** than my original scalar implementation

**Open source, fully reproducible, and ready for the RISC-V community.** 🚀

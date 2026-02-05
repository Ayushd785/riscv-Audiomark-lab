# =============================================================================
# RISC-V Hybrid Port CMake Configuration
# =============================================================================
# Strategy: "Fast Vector Math" + "Safe Scalar Neural Network"
#   - RVV intrinsics for DSP functions (hand-written, LMUL=8)
#   - Pure scalar C for Neural Network (prevents -35dB SNR validation failures)
# =============================================================================

include_directories(${PORT_DIR})

# -----------------------------------------------------------------------------
# Compiler Flags for Hybrid Strategy
# -----------------------------------------------------------------------------
# -march=rv64gcv    : Enable RISC-V Vector 1.0 extensions for manual RVV code
# -mabi=lp64d       : LP64 ABI with double-precision float
# -O3               : Maximum optimization
# -fno-tree-vectorize : CRITICAL! Disable auto-vectorization to prevent
#                       compiler from "optimizing" scalar NN code into broken
#                       vector code that causes segfaults and SNR failures
# -Wno-implicit-function-declaration : Suppress legacy upstream code warnings
# -----------------------------------------------------------------------------
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d -O3 -fno-tree-vectorize -Wno-error=implicit-function-declaration -Wno-implicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv -mabi=lp64d -O3 -fno-tree-vectorize")

# -----------------------------------------------------------------------------
# FFT Configuration
# -----------------------------------------------------------------------------
# Use SMALLFT for SpeexDSP FFT (built-in, no external dependencies)
add_definitions(-DUSE_SMALLFT)

# -----------------------------------------------------------------------------
# Port Source Files
# -----------------------------------------------------------------------------
set(PORT_SOURCE
    ${PORT_DIR}/th_api.c
)

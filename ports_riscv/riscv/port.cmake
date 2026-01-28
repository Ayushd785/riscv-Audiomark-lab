# RISC-V Vector Port CMake Configuration

include_directories(${PORT_DIR})

# Add RVV compiler flags
# -Wno-error=implicit-function-declaration: Work around upstream AudioMark missing includes
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d -Wno-error=implicit-function-declaration -Wno-implicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv -mabi=lp64d")

# Use SMALLFT for SpeexDSP FFT (built-in, no external deps)
add_definitions(-DUSE_SMALLFT)

# Port source files
set(PORT_SOURCE
    ${PORT_DIR}/th_api.c
)

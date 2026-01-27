# riscv_toolchain.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Point to the sysroot INSIDE the Docker container
set(CMAKE_SYSROOT /opt/riscv/sysroot)

# Define the compilers
set(CMAKE_C_COMPILER /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/riscv/bin/riscv64-unknown-linux-gnu-g++)

# Don't check for full compilation (since we are cross-compiling)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Flags to ensure sysroot is found
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=${CMAKE_SYSROOT}")

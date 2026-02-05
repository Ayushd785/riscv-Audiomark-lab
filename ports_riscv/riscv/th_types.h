/**
 * =============================================================================
 * RISC-V Hybrid Port Type Definitions
 * =============================================================================
 * Defines FFT and Matrix structs required by the AudioMark benchmark.
 * These types are used by both vectorized DSP functions and scalar NN code.
 * =============================================================================
 */

#ifndef __TH_TYPES_H
#define __TH_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * Floating Point Type
 * ============================================================================
 */
#define TH_FLOAT32_TYPE float

/* ============================================================================
 * Complex FFT Instance
 * ============================================================================
 * Used for complex-to-complex FFT operations.
 * The cfg pointer holds KissFFT or SpeexDSP internal state.
 * ============================================================================
 */
typedef struct {
  int fft_length; /* FFT size (e.g., 256, 512) */
  void *cfg;      /* Pointer to FFT library state (KissFFT/SpeexDSP) */
  int inverse;    /* 0 = forward FFT, 1 = inverse FFT */
} th_cfft_instance_f32;

/* ============================================================================
 * Real FFT Instance
 * ============================================================================
 * Used for real-to-complex and complex-to-real FFT operations.
 * Separate configs for forward and inverse transforms.
 * ============================================================================
 */
typedef struct {
  int fft_length; /* FFT size (e.g., 256, 512) */
  void *cfg_fwd;  /* Pointer to forward FFT config */
  void *cfg_inv;  /* Pointer to inverse FFT config */
} th_rfft_instance_f32;

/* ============================================================================
 * Matrix Instance
 * ============================================================================
 * Used for matrix-vector multiplication in DSP operations.
 * ============================================================================
 */
typedef struct {
  uint32_t numRows; /* Number of rows in the matrix */
  uint32_t numCols; /* Number of columns in the matrix */
  float *pData;     /* Pointer to matrix data (row-major order) */
} th_matrix_instance_f32;

/* ============================================================================
 * AudioMark Type Mappings
 * ============================================================================
 * These macros map AudioMark's expected type names to our struct definitions.
 * ============================================================================
 */
#define TH_MATRIX_INSTANCE_FLOAT32_TYPE th_matrix_instance_f32
#define TH_RFFT_INSTANCE_FLOAT32_TYPE th_rfft_instance_f32
#define TH_CFFT_INSTANCE_FLOAT32_TYPE th_cfft_instance_f32

#endif /* __TH_TYPES_H */

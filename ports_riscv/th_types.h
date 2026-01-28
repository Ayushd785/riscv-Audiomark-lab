/**
 * RISC-V Vector Port Type Definitions
 */

#ifndef __TH_TYPES_H
#define __TH_TYPES_H

#include <stdint.h>
#include <stddef.h>

#define TH_FLOAT32_TYPE float

// Simple struct-based types for FFT and Matrix
// These are placeholders - full FFT support requires external library

typedef struct
{
    int fft_length;
} th_cfft_instance_f32;

typedef struct
{
    int fft_length;
} th_rfft_instance_f32;

typedef struct
{
    uint32_t numRows;
    uint32_t numCols;
    float   *pData;
} th_matrix_instance_f32;

#define TH_MATRIX_INSTANCE_FLOAT32_TYPE th_matrix_instance_f32
#define TH_RFFT_INSTANCE_FLOAT32_TYPE   th_rfft_instance_f32
#define TH_CFFT_INSTANCE_FLOAT32_TYPE   th_cfft_instance_f32

#endif /* __TH_TYPES_H */

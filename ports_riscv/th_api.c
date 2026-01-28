/**
 * RISC-V Vector (RVV) Port for EEMBC AudioMark
 *
 * This port implements DSP functions using RISC-V Vector intrinsics.
 * Optimized for RVV 1.0 specification.
 */

#include "ee_audiomark.h"
#include "ee_api.h"
#include <riscv_vector.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// Audio Data Buffers (Required by AudioMark)
// ============================================================================

const int16_t downlink_audio[NINPUT_SAMPLES] = {
#include "ee_data/noise.txt"
};
const int16_t left_microphone_capture[NINPUT_SAMPLES] = {
#include "ee_data/left0.txt"
};
const int16_t right_microphone_capture[NINPUT_SAMPLES] = {
#include "ee_data/right0.txt"
};
int16_t for_asr[NINPUT_SAMPLES];

// Inter-component buffers
int16_t audio_input[SAMPLES_PER_AUDIO_FRAME];
int16_t left_capture[SAMPLES_PER_AUDIO_FRAME];
int16_t right_capture[SAMPLES_PER_AUDIO_FRAME];
int16_t beamformer_output[SAMPLES_PER_AUDIO_FRAME];
int16_t aec_output[SAMPLES_PER_AUDIO_FRAME];
int16_t audio_fifo[AUDIO_FIFO_SAMPLES];
int8_t  mfcc_fifo[MFCC_FIFO_BYTES];
int8_t  classes[OUT_DIM];

// ============================================================================
// Memory Functions
// ============================================================================

void *
th_malloc(size_t size, int req)
{
    (void)req;
    return malloc(size);
}

void
th_free(void *mem, int req)
{
    (void)req;
    free(mem);
}

void *
th_memcpy(void *restrict dst, const void *restrict src, size_t n)
{
    return memcpy(dst, src, n);
}

void *
th_memmove(void *dst, const void *src, size_t n)
{
    return memmove(dst, src, n);
}

void *
th_memset(void *b, int c, size_t len)
{
    return memset(b, c, len);
}

// ============================================================================
// FFT Functions (Stub - requires external FFT library)
// ============================================================================

ee_status_t
th_cfft_init_f32(ee_cfft_f32_t *p_instance, int fft_length)
{
    (void)p_instance;
    (void)fft_length;
    return EE_STATUS_OK;
}

void
th_cfft_f32(ee_cfft_f32_t *p_instance,
            ee_f32_t      *p_buf,
            uint8_t        ifftFlag,
            uint8_t        bitReverseFlagR)
{
    (void)p_instance;
    (void)p_buf;
    (void)ifftFlag;
    (void)bitReverseFlagR;
}

ee_status_t
th_rfft_init_f32(ee_rfft_f32_t *p_instance, int fft_length)
{
    (void)p_instance;
    (void)fft_length;
    return EE_STATUS_OK;
}

void
th_rfft_f32(ee_rfft_f32_t *p_instance,
            ee_f32_t      *p_in,
            ee_f32_t      *p_out,
            uint8_t        ifftFlag)
{
    (void)p_instance;
    (void)p_in;
    (void)p_out;
    (void)ifftFlag;
}

// ============================================================================
// RVV-Optimized DSP Functions
// ============================================================================

/**
 * Vector Addition: z = x + y
 * Strategy: LMUL=8 for maximum throughput
 */
void
th_add_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    const float *y = p_b;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += vl, y += vl, z += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        vfloat32m8_t vz = __riscv_vfadd_vv_f32m8(vx, vy, vl);
        __riscv_vse32_v_f32m8(z, vz, vl);
    }
}

/**
 * Vector Subtraction: z = x - y
 * Strategy: LMUL=8 for maximum throughput
 */
void
th_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    const float *y = p_b;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += vl, y += vl, z += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        vfloat32m8_t vz = __riscv_vfsub_vv_f32m8(vx, vy, vl);
        __riscv_vse32_v_f32m8(z, vz, vl);
    }
}

/**
 * Vector Multiply: z = x * y
 * Strategy: LMUL=8 for maximum throughput
 */
void
th_multiply_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    const float *y = p_b;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += vl, y += vl, z += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vx, vy, vl);
        __riscv_vse32_v_f32m8(z, vz, vl);
    }
}

/**
 * Dot Product (Reduction)
 * Strategy: Accumulate in vector, reduce once at end
 */
void
th_dot_prod_f32(ee_f32_t *p_a, ee_f32_t *p_b, uint32_t len, ee_f32_t *p_result)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    const float *y = p_b;

    // Initialize accumulator
    vfloat32m8_t v_sum
        = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

    for (; n > 0; n -= vl, x += vl, y += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        v_sum           = __riscv_vfmacc_vv_f32m8(v_sum, vx, vy, vl);
    }

    // Horizontal reduction
    vfloat32m1_t v_zero
        = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t v_res = __riscv_vfredusum_vs_f32m8_f32m1(
        v_sum, v_zero, __riscv_vsetvlmax_e32m8());
    *p_result = __riscv_vfmv_f_s_f32m1_f32(v_res);
}

/**
 * Complex Multiply (Segmented Load/Store)
 * Strategy: vlseg2 for automatic de-interleaving, LMUL=4
 */
void
th_cmplx_mult_cmplx_f32(const ee_f32_t *p_a,
                        const ee_f32_t *p_b,
                        ee_f32_t       *p_c,
                        uint32_t        len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    const float *y = p_b;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += 2 * vl, y += 2 * vl, z += 2 * vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        // De-interleave load
        vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(x, vl);
        vfloat32m4x2_t vy = __riscv_vlseg2e32_v_f32m4x2(y, vl);

        vfloat32m4_t xr = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
        vfloat32m4_t xi = __riscv_vget_v_f32m4x2_f32m4(vx, 1);
        vfloat32m4_t yr = __riscv_vget_v_f32m4x2_f32m4(vy, 0);
        vfloat32m4_t yi = __riscv_vget_v_f32m4x2_f32m4(vy, 1);

        // (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        vfloat32m4_t zr = __riscv_vfmul_vv_f32m4(xr, yr, vl);
        zr = __riscv_vfnmsac_vv_f32m4(zr, xi, yi, vl); // zr = zr - xi*yi

        vfloat32m4_t zi = __riscv_vfmul_vv_f32m4(xr, yi, vl);
        zi = __riscv_vfmacc_vv_f32m4(zi, xi, yr, vl); // zi = zi + xi*yr

        // Interleave store
        vfloat32m4x2_t vz;
        vz = __riscv_vset_v_f32m4_f32m4x2(vz, 0, zr);
        vz = __riscv_vset_v_f32m4_f32m4x2(vz, 1, zi);
        __riscv_vsseg2e32_v_f32m4x2(z, vz, vl);
    }
}

/**
 * Complex Conjugate: conj(a+bi) = a-bi
 */
void
th_cmplx_conj_f32(const ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += 2 * vl, z += 2 * vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(x, vl);
        vfloat32m4_t   xr = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
        vfloat32m4_t   xi = __riscv_vget_v_f32m4x2_f32m4(vx, 1);

        // Negate imaginary part
        vfloat32m4_t zi = __riscv_vfneg_v_f32m4(xi, vl);

        vfloat32m4x2_t vz;
        vz = __riscv_vset_v_f32m4_f32m4x2(vz, 0, xr);
        vz = __riscv_vset_v_f32m4_f32m4x2(vz, 1, zi);
        __riscv_vsseg2e32_v_f32m4x2(z, vz, vl);
    }
}

/**
 * Complex Dot Product
 * Returns: sum(a * conj(b)) = (real_sum, imag_sum)
 */
void
th_cmplx_dot_prod_f32(const ee_f32_t *p_a,
                      const ee_f32_t *p_b,
                      uint32_t        len,
                      ee_f32_t       *p_r,
                      ee_f32_t       *p_i)
{
    size_t       vl;
    int          n = (int)len;
    const float *a = p_a;
    const float *b = p_b;

    vfloat32m4_t sum_r
        = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t sum_i
        = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

    for (; n > 0; n -= vl, a += 2 * vl, b += 2 * vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        vfloat32m4x2_t va = __riscv_vlseg2e32_v_f32m4x2(a, vl);
        vfloat32m4x2_t vb = __riscv_vlseg2e32_v_f32m4x2(b, vl);

        vfloat32m4_t ar = __riscv_vget_v_f32m4x2_f32m4(va, 0);
        vfloat32m4_t ai = __riscv_vget_v_f32m4x2_f32m4(va, 1);
        vfloat32m4_t br = __riscv_vget_v_f32m4x2_f32m4(vb, 0);
        vfloat32m4_t bi = __riscv_vget_v_f32m4x2_f32m4(vb, 1);

        // Real: ar*br + ai*bi
        sum_r = __riscv_vfmacc_vv_f32m4(sum_r, ar, br, vl);
        sum_r = __riscv_vfmacc_vv_f32m4(sum_r, ai, bi, vl);

        // Imag: ai*br - ar*bi
        sum_i = __riscv_vfmacc_vv_f32m4(sum_i, ai, br, vl);
        sum_i = __riscv_vfnmsac_vv_f32m4(sum_i, ar, bi, vl);
    }

    // Reduce
    vfloat32m1_t v_zero
        = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t res_r = __riscv_vfredusum_vs_f32m4_f32m1(
        sum_r, v_zero, __riscv_vsetvlmax_e32m4());
    vfloat32m1_t res_i = __riscv_vfredusum_vs_f32m4_f32m1(
        sum_i, v_zero, __riscv_vsetvlmax_e32m4());

    *p_r = __riscv_vfmv_f_s_f32m1_f32(res_r);
    *p_i = __riscv_vfmv_f_s_f32m1_f32(res_i);
}

/**
 * Complex Magnitude: |a+bi| = sqrt(a^2 + b^2)
 */
void
th_cmplx_mag_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += 2 * vl, z += vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(x, vl);
        vfloat32m4_t   xr = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
        vfloat32m4_t   xi = __riscv_vget_v_f32m4x2_f32m4(vx, 1);

        // mag^2 = real^2 + imag^2
        vfloat32m4_t mag_sq = __riscv_vfmul_vv_f32m4(xr, xr, vl);
        mag_sq              = __riscv_vfmacc_vv_f32m4(mag_sq, xi, xi, vl);

        // sqrt
        vfloat32m4_t mag = __riscv_vfsqrt_v_f32m4(mag_sq, vl);
        __riscv_vse32_v_f32m4(z, mag, vl);
    }
}

/**
 * Absolute Maximum with Index
 */
void
th_absmax_f32(const ee_f32_t *p_in,
              uint32_t        len,
              ee_f32_t       *p_max,
              uint32_t       *p_index)
{
    float    max_val = 0.0f;
    uint32_t max_idx = 0;

    for (uint32_t i = 0; i < len; i++)
    {
        float abs_val = fabsf(p_in[i]);
        if (abs_val > max_val)
        {
            max_val = abs_val;
            max_idx = i;
        }
    }
    *p_max   = max_val;
    *p_index = max_idx;
}

/**
 * Vector Offset: z = x + offset
 */
void
th_offset_f32(ee_f32_t *p_a, ee_f32_t offset, ee_f32_t *p_c, uint32_t len)
{
    size_t       vl;
    int          n = (int)len;
    const float *x = p_a;
    float       *z = p_c;

    for (; n > 0; n -= vl, x += vl, z += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vz = __riscv_vfadd_vf_f32m8(vx, offset, vl);
        __riscv_vse32_v_f32m8(z, vz, vl);
    }
}

/**
 * Vector Log (Element-wise natural log)
 */
void
th_vlog_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len)
{
    // Scalar fallback - RVV doesn't have native log
    for (uint32_t i = 0; i < len; i++)
    {
        p_c[i] = logf(p_a[i]);
    }
}

/**
 * Int16 to Float32 conversion
 */
void
th_int16_to_f32(const int16_t *p_src, ee_f32_t *p_dst, uint32_t len)
{
    size_t         vl;
    int            n   = (int)len;
    const int16_t *src = p_src;
    float         *dst = p_dst;

    for (; n > 0; n -= vl, src += vl, dst += vl)
    {
        vl              = __riscv_vsetvl_e16m4(n);
        vint16m4_t   vs = __riscv_vle16_v_i16m4(src, vl);
        vint32m8_t   vi = __riscv_vwcvt_x_x_v_i32m8(vs, vl);
        vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vi, vl);
        // Normalize to Q15 range
        vf = __riscv_vfdiv_vf_f32m8(vf, 32768.0f, vl);
        __riscv_vse32_v_f32m8(dst, vf, vl);
    }
}

/**
 * Float32 to Int16 conversion
 */
void
th_f32_to_int16(const ee_f32_t *p_src, int16_t *p_dst, uint32_t len)
{
    size_t       vl;
    int          n   = (int)len;
    const float *src = p_src;
    int16_t     *dst = p_dst;

    for (; n > 0; n -= vl, src += vl, dst += vl)
    {
        vl              = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vf = __riscv_vle32_v_f32m8(src, vl);
        // Scale from normalized to Q15
        vf            = __riscv_vfmul_vf_f32m8(vf, 32768.0f, vl);
        vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
        vint16m4_t vs = __riscv_vncvt_x_x_w_i16m4(vi, vl);
        __riscv_vse16_v_i16m4(dst, vs, vl);
    }
}

/**
 * Matrix-Vector Multiplication
 */
void
th_mat_vec_mult_f32(ee_matrix_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c)
{
    uint32_t rows = p_a->numRows;
    uint32_t cols = p_a->numCols;
    float   *mat  = p_a->pData;

    for (uint32_t i = 0; i < rows; i++)
    {
        th_dot_prod_f32(&mat[i * cols], p_b, cols, &p_c[i]);
    }
}

// ============================================================================
// Neural Network Functions (Stub - requires TFLite Micro or similar)
// ============================================================================

void
th_nn_init(void)
{
    // Placeholder for NN initialization
}

ee_status_t
th_nn_classify(const int8_t in_data[490], int8_t out_data[12])
{
    (void)in_data;
    (void)out_data;
    return EE_STATUS_OK;
}

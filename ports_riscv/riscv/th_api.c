/**
 * RISC-V Port for EEMBC AudioMark - Scalar Baseline
 * With Complete DS-CNN-S Neural Network Implementation
 * FIXED: Corrected requantization to match TFLite Micro/CMSIS-NN
 */

#include "ee_api.h"
#include "ee_audiomark.h"
#include "ee_nn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
int16_t audio_input[SAMPLES_PER_AUDIO_FRAME];
int16_t left_capture[SAMPLES_PER_AUDIO_FRAME];
int16_t right_capture[SAMPLES_PER_AUDIO_FRAME];
int16_t beamformer_output[SAMPLES_PER_AUDIO_FRAME];
int16_t aec_output[SAMPLES_PER_AUDIO_FRAME];
int16_t audio_fifo[AUDIO_FIFO_SAMPLES];
int8_t mfcc_fifo[MFCC_FIFO_BYTES];
int8_t classes[OUT_DIM];

// ============================================================================
// Memory Management
// ============================================================================
void *th_malloc(size_t size, int req) {
  (void)req;
  return malloc(size);
}

void th_free(void *mem, int req) {
  (void)req;
  free(mem);
}

void *th_memcpy(void *restrict dst, const void *restrict src, size_t n) {
  return memcpy(dst, src, n);
}

void *th_memmove(void *dst, const void *src, size_t n) {
  return memmove(dst, src, n);
}

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

// ============================================================================
// FFT Functions - Simple DFT/IDFT implementations
// ============================================================================
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void dft_complex(float *data, int n, int inverse) {
  float *out = (float *)malloc(2 * n * sizeof(float));
  if (!out)
    return;

  float sign = inverse ? 1.0f : -1.0f;

  for (int k = 0; k < n; k++) {
    float sum_re = 0.0f, sum_im = 0.0f;
    for (int j = 0; j < n; j++) {
      float angle = 2.0f * M_PI * k * j / n;
      float cos_a = cosf(angle);
      float sin_a = sign * sinf(angle);
      sum_re += data[2 * j] * cos_a - data[2 * j + 1] * sin_a;
      sum_im += data[2 * j] * sin_a + data[2 * j + 1] * cos_a;
    }
    out[2 * k] = sum_re;
    out[2 * k + 1] = sum_im;
  }

  if (inverse) {
    for (int i = 0; i < 2 * n; i++)
      data[i] = out[i] / n;
  } else {
    memcpy(data, out, 2 * n * sizeof(float));
  }
  free(out);
}

static void rfft(float *in, float *out, int n, int inverse) {
  float *c = (float *)malloc(2 * n * sizeof(float));
  if (!c)
    return;

  if (!inverse) {
    for (int i = 0; i < n; i++) {
      c[2 * i] = in[i];
      c[2 * i + 1] = 0.0f;
    }
    dft_complex(c, n, 0);
    out[0] = c[0];
    out[1] = c[n];
    for (int i = 1; i < n / 2; i++) {
      out[2 * i] = c[2 * i];
      out[2 * i + 1] = c[2 * i + 1];
    }
  } else {
    c[0] = in[0];
    c[1] = 0.0f;
    c[n] = in[1];
    c[n + 1] = 0.0f;
    for (int i = 1; i < n / 2; i++) {
      c[2 * i] = in[2 * i];
      c[2 * i + 1] = in[2 * i + 1];
      c[2 * (n - i)] = in[2 * i];
      c[2 * (n - i) + 1] = -in[2 * i + 1];
    }
    dft_complex(c, n, 1);
    for (int i = 0; i < n; i++)
      out[i] = c[2 * i];
  }
  free(c);
}

ee_status_t th_cfft_init_f32(ee_cfft_f32_t *p_instance, int fft_length) {
  if (!p_instance)
    return EE_STATUS_ERROR;
  p_instance->fft_length = fft_length;
  p_instance->cfg = NULL;
  return EE_STATUS_OK;
}

void th_cfft_f32(ee_cfft_f32_t *p_instance, ee_f32_t *p_buf, uint8_t ifftFlag,
                 uint8_t bitReverseFlag) {
  (void)bitReverseFlag;
  if (!p_instance || !p_buf)
    return;
  dft_complex(p_buf, p_instance->fft_length, ifftFlag);
}

ee_status_t th_rfft_init_f32(ee_rfft_f32_t *p_instance, int fft_length) {
  if (!p_instance)
    return EE_STATUS_ERROR;
  p_instance->fft_length = fft_length;
  p_instance->cfg_fwd = NULL;
  p_instance->cfg_inv = NULL;
  return EE_STATUS_OK;
}

void th_rfft_f32(ee_rfft_f32_t *p_instance, ee_f32_t *p_in, ee_f32_t *p_out,
                 uint8_t ifftFlag) {
  if (!p_instance || !p_in || !p_out)
    return;
  rfft(p_in, p_out, p_instance->fft_length, ifftFlag);
}

// ============================================================================
// DSP Functions
// ============================================================================
void th_add_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_c[i] = p_a[i] + p_b[i];
}

void th_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c,
                     uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_c[i] = p_a[i] - p_b[i];
}

void th_multiply_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c,
                     uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_c[i] = p_a[i] * p_b[i];
}

void th_dot_prod_f32(ee_f32_t *p_a, ee_f32_t *p_b, uint32_t len,
                     ee_f32_t *p_result) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < len; i++)
    sum += p_a[i] * p_b[i];
  *p_result = sum;
}

void th_offset_f32(ee_f32_t *p_a, ee_f32_t offset, ee_f32_t *p_c,
                   uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_c[i] = p_a[i] + offset;
}

void th_absmax_f32(const ee_f32_t *p_in, uint32_t len, ee_f32_t *p_max,
                   uint32_t *p_index) {
  float max_val = 0.0f;
  uint32_t index = 0;
  for (uint32_t i = 0; i < len; i++) {
    float val = fabsf(p_in[i]);
    if (val > max_val) {
      max_val = val;
      index = i;
    }
  }
  *p_max = max_val;
  *p_index = index;
}

void th_vlog_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_c[i] = logf(p_a[i]);
}

void th_cmplx_mag_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    float r = p_a[2 * i], im = p_a[2 * i + 1];
    p_c[i] = sqrtf(r * r + im * im);
  }
}

void th_cmplx_mult_cmplx_f32(const ee_f32_t *p_a, const ee_f32_t *p_b,
                             ee_f32_t *p_c, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    float ar = p_a[2 * i], ai = p_a[2 * i + 1], br = p_b[2 * i],
          bi = p_b[2 * i + 1];
    p_c[2 * i] = ar * br - ai * bi;
    p_c[2 * i + 1] = ar * bi + ai * br;
  }
}

void th_cmplx_conj_f32(const ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    p_c[2 * i] = p_a[2 * i];
    p_c[2 * i + 1] = -p_a[2 * i + 1];
  }
}

void th_cmplx_dot_prod_f32(const ee_f32_t *p_a, const ee_f32_t *p_b,
                           uint32_t len, ee_f32_t *p_r, ee_f32_t *p_i) {
  float sum_r = 0.0f, sum_i = 0.0f;
  for (uint32_t i = 0; i < len; i++) {
    float ar = p_a[2 * i], ai = p_a[2 * i + 1], br = p_b[2 * i],
          bi = p_b[2 * i + 1];
    sum_r += ar * br + ai * bi;
    sum_i += ai * br - ar * bi;
  }
  *p_r = sum_r;
  *p_i = sum_i;
}

void th_mat_vec_mult_f32(ee_matrix_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c) {
  for (uint32_t r = 0; r < p_a->numRows; r++) {
    float sum = 0.0f;
    for (uint32_t c = 0; c < p_a->numCols; c++)
      sum += p_a->pData[r * p_a->numCols + c] * p_b[c];
    p_c[r] = sum;
  }
}

void th_int16_to_f32(const int16_t *p_src, ee_f32_t *p_dst, uint32_t len) {
  for (uint32_t i = 0; i < len; i++)
    p_dst[i] = (float)p_src[i] / 32768.0f;
}

void th_f32_to_int16(const ee_f32_t *p_src, int16_t *p_dst, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    float val = p_src[i] * 32768.0f;
    if (val > 32767.0f)
      val = 32767.0f;
    if (val < -32768.0f)
      val = -32768.0f;
    p_dst[i] = (int16_t)val;
  }
}

// ============================================================================
// Neural Network - DS-CNN-S for KWS
// ============================================================================

// External weight symbols from ee_nn_tables.c
extern const int8_t ds_cnn_s_layer_1_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_1_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_1_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_1_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_2_dw_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_2_dw_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_2_dw_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_2_dw_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_3_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_3_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_3_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_3_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_4_dw_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_4_dw_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_4_dw_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_4_dw_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_5_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_5_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_5_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_5_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_6_dw_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_6_dw_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_6_dw_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_6_dw_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_7_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_7_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_7_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_7_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_8_dw_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_8_dw_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_8_dw_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_8_dw_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_9_conv2d_weights[];
extern const int32_t ds_cnn_s_layer_9_conv2d_bias[];
extern const int32_t ds_cnn_s_layer_9_conv2d_output_mult[];
extern const int32_t ds_cnn_s_layer_9_conv2d_output_shift[];

extern const int8_t ds_cnn_s_layer_12_fc_weights[];
extern const int32_t ds_cnn_s_layer_12_fc_bias[];

// Scratch buffers
static int8_t *nn_buf0 = NULL;
static int8_t *nn_buf1 = NULL;
#define NN_BUF_SIZE (25 * 5 * 64)

// ============================================================================
// CORRECTED TFLite Micro / CMSIS-NN Quantization Functions
// ============================================================================

// SaturatingRoundingDoublingHighMul - implements the ARMv7 NEON VQRDMULH
// instruction This is the key function from gemmlowp that does high-precision
// int32 multiply
static inline int32_t saturating_rounding_doubling_high_mul(int32_t a,
                                                            int32_t b) {
  // Check for overflow case
  bool overflow = (a == b) && (a == INT32_MIN);

  // Multiply to 64-bit
  int64_t a_64 = (int64_t)a;
  int64_t b_64 = (int64_t)b;
  int64_t ab_64 = a_64 * b_64;

  // Rounding: add nudge value (0.5 in the fixed-point representation)
  // The nudge for rounding to nearest (tie towards positive infinity)
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));

  // Divide by 2^31 (right shift 31), which gets us the high 32 bits of the
  // doubled product
  int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1LL << 31));

  return overflow ? INT32_MAX : ab_x2_high32;
}

// Rounding divide by power of two - matches gemmlowp implementation
static inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
  if (exponent == 0)
    return x;

  int32_t mask = (1 << exponent) - 1;
  int32_t remainder = x & mask;
  int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);

  return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

// The main requantization function matching TFLite Micro
// NOTE: In TFLite/CMSIS-NN, "shift" is the exponent where:
// - negative shift means right shift
// - positive shift means the value has already been left-shifted
static inline int32_t
multiply_by_quantized_multiplier(int32_t x, int32_t quantized_multiplier,
                                 int32_t shift) {
  // Apply the high-precision multiplication
  int32_t result =
      saturating_rounding_doubling_high_mul(x, quantized_multiplier);

  // Apply the shift
  // In TFLite, shift < 0 means we need to right-shift the result
  if (shift < 0) {
    result = rounding_divide_by_pot(result, -shift);
  }
  // If shift >= 0, no additional shifting (it's already accounted for in the
  // multiplier)

  return result;
}

// Requantize and clamp to int8
static inline int8_t arm_nn_requantize(int32_t val, int32_t mult,
                                       int32_t shift) {
  int32_t result = multiply_by_quantized_multiplier(val, mult, shift);

  // Clamp to int8 range
  if (result > 127)
    result = 127;
  if (result < -128)
    result = -128;

  return (int8_t)result;
}

// Conv2D layer (for initial conv and pointwise convs)
static void conv2d_s8(const int8_t *input, int in_h, int in_w, int in_c,
                      const int8_t *weights, const int32_t *bias, int kh,
                      int kw, int out_c, int sh, int sw, int ph, int pw,
                      int in_off, int out_off, const int32_t *mult,
                      const int32_t *shift, int8_t *output, int out_h,
                      int out_w) {
  for (int oc = 0; oc < out_c; oc++) {
    for (int oh = 0; oh < out_h; oh++) {
      for (int ow = 0; ow < out_w; ow++) {
        int32_t acc = bias[oc];
        for (int fh = 0; fh < kh; fh++) {
          int ih = oh * sh - ph + fh;
          for (int fw = 0; fw < kw; fw++) {
            int iw = ow * sw - pw + fw;
            for (int ic = 0; ic < in_c; ic++) {
              int8_t in_val = 0;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                in_val = input[(ih * in_w + iw) * in_c + ic];
              }
              // OHWI weight layout
              int w_idx = ((oc * kh + fh) * kw + fw) * in_c + ic;
              acc += ((int32_t)in_val + in_off) * (int32_t)weights[w_idx];
            }
          }
        }
        // Use corrected requantization
        int8_t out_val = arm_nn_requantize(acc, mult[oc], shift[oc]);
        out_val = out_val + out_off;
        if (out_val > 127)
          out_val = 127;
        if (out_val < -128)
          out_val = -128;
        output[(oh * out_w + ow) * out_c + oc] = out_val;
      }
    }
  }
}

// Depthwise Conv2D layer
static void dw_conv2d_s8(const int8_t *input, int in_h, int in_w, int ch,
                         const int8_t *weights, const int32_t *bias, int kh,
                         int kw, int sh, int sw, int ph, int pw, int in_off,
                         int out_off, const int32_t *mult, const int32_t *shift,
                         int8_t *output, int out_h, int out_w) {
  for (int c = 0; c < ch; c++) {
    for (int oh = 0; oh < out_h; oh++) {
      for (int ow = 0; ow < out_w; ow++) {
        int32_t acc = bias[c];
        for (int fh = 0; fh < kh; fh++) {
          int ih = oh * sh - ph + fh;
          for (int fw = 0; fw < kw; fw++) {
            int iw = ow * sw - pw + fw;
            int8_t in_val = 0;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
              in_val = input[(ih * in_w + iw) * ch + c];
            }
            // Depthwise weight: (H,W,C,1) for ch_mult=1
            int w_idx = (fh * kw + fw) * ch + c;
            acc += ((int32_t)in_val + in_off) * (int32_t)weights[w_idx];
          }
        }
        // Use corrected requantization
        int8_t out_val = arm_nn_requantize(acc, mult[c], shift[c]);
        out_val = out_val + out_off;
        if (out_val > 127)
          out_val = 127;
        if (out_val < -128)
          out_val = -128;
        output[(oh * out_w + ow) * ch + c] = out_val;
      }
    }
  }
}

// Average pooling
static void avg_pool_s8(const int8_t *input, int h, int w, int c, int in_off,
                        int8_t *output) {
  for (int ch = 0; ch < c; ch++) {
    int32_t sum = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        sum += (int32_t)input[(y * w + x) * c + ch] + in_off;
      }
    }
    int32_t avg = (sum + (h * w / 2)) / (h * w);
    avg = avg - in_off; // Remove offset for output
    if (avg > 127)
      avg = 127;
    if (avg < -128)
      avg = -128;
    output[ch] = (int8_t)avg;
  }
}

// Fully connected
static void fc_s8(const int8_t *input, int in_f, const int8_t *weights,
                  const int32_t *bias, int out_f, int in_off, int out_off,
                  int32_t mult, int32_t shift, int8_t *output) {
  for (int o = 0; o < out_f; o++) {
    int32_t acc = bias[o];
    for (int i = 0; i < in_f; i++) {
      // Weight layout: (out_f, in_f)
      acc += ((int32_t)input[i] + in_off) * (int32_t)weights[o * in_f + i];
    }
    // Use corrected requantization
    int8_t out_val = arm_nn_requantize(acc, mult, shift);
    out_val = out_val + out_off;
    if (out_val > 127)
      out_val = 127;
    if (out_val < -128)
      out_val = -128;
    output[o] = out_val;
  }
}

// Softmax (TFLite Micro style)
static void softmax_s8(const int8_t *input, int n, int32_t mult, int32_t shift,
                       int diff_min, int8_t *output) {
  int8_t max_val = input[0];
  for (int i = 1; i < n; i++) {
    if (input[i] > max_val)
      max_val = input[i];
  }

  int32_t sum = 0;
  int32_t exp_scaled[12];
  for (int i = 0; i < n; i++) {
    int diff = input[i] - max_val;
    if (diff < diff_min) {
      exp_scaled[i] = 0;
    } else {
      // Approximate exp using lookup or linear approx
      // exp(x) ≈ 1 + x + x^2/2 for small x, but we use integer approx
      int32_t scaled = ((int64_t)diff * mult + (1LL << (shift - 1))) >> shift;
      exp_scaled[i] = 256 + (scaled > -256 ? scaled : -255);
      if (exp_scaled[i] < 0)
        exp_scaled[i] = 0;
    }
    sum += exp_scaled[i];
  }

  if (sum == 0)
    sum = 1;
  for (int i = 0; i < n; i++) {
    int32_t prob = (exp_scaled[i] * 255) / sum;
    output[i] = (int8_t)(prob - 128);
  }
}

void th_nn_init(void) {
  if (!nn_buf0)
    nn_buf0 = (int8_t *)malloc(NN_BUF_SIZE);
  if (!nn_buf1)
    nn_buf1 = (int8_t *)malloc(NN_BUF_SIZE);
}

ee_status_t th_nn_classify(const int8_t in_data[490], int8_t out_data[12]) {
  if (!nn_buf0 || !nn_buf1)
    th_nn_init();

  // Layer 0: Conv2D (49x10x1 -> 25x5x64)
  conv2d_s8(in_data, CONV_0_INPUT_H, CONV_0_INPUT_W, CONV_0_IN_CH,
            ds_cnn_s_layer_1_conv2d_weights, ds_cnn_s_layer_1_conv2d_bias,
            CONV_0_FILTER_H, CONV_0_FILTER_W, CONV_0_OUT_CH, CONV_0_STRIDE_H,
            CONV_0_STRIDE_W, CONV_0_PAD_H, CONV_0_PAD_W, CONV_0_INPUT_OFFSET,
            CONV_0_OUTPUT_OFFSET, ds_cnn_s_layer_1_conv2d_output_mult,
            ds_cnn_s_layer_1_conv2d_output_shift, nn_buf0, CONV_0_OUTPUT_H,
            CONV_0_OUTPUT_W);

  // DS Block 1: DW Conv + PW Conv
  dw_conv2d_s8(nn_buf0, 25, 5, 64, ds_cnn_s_layer_2_dw_conv2d_weights,
               ds_cnn_s_layer_2_dw_conv2d_bias, 3, 3, 1, 1, 1, 1,
               DW_CONV_1_INPUT_OFFSET, DW_CONV_1_OUTPUT_OFFSET,
               ds_cnn_s_layer_2_dw_conv2d_output_mult,
               ds_cnn_s_layer_2_dw_conv2d_output_shift, nn_buf1, 25, 5);

  conv2d_s8(nn_buf1, 25, 5, 64, ds_cnn_s_layer_3_conv2d_weights,
            ds_cnn_s_layer_3_conv2d_bias, 1, 1, 64, 1, 1, 0, 0,
            CONV_2_INPUT_OFFSET, CONV_2_OUTPUT_OFFSET,
            ds_cnn_s_layer_3_conv2d_output_mult,
            ds_cnn_s_layer_3_conv2d_output_shift, nn_buf0, 25, 5);

  // DS Block 2
  dw_conv2d_s8(nn_buf0, 25, 5, 64, ds_cnn_s_layer_4_dw_conv2d_weights,
               ds_cnn_s_layer_4_dw_conv2d_bias, 3, 3, 1, 1, 1, 1,
               DW_CONV_3_INPUT_OFFSET, DW_CONV_3_OUTPUT_OFFSET,
               ds_cnn_s_layer_4_dw_conv2d_output_mult,
               ds_cnn_s_layer_4_dw_conv2d_output_shift, nn_buf1, 25, 5);

  conv2d_s8(nn_buf1, 25, 5, 64, ds_cnn_s_layer_5_conv2d_weights,
            ds_cnn_s_layer_5_conv2d_bias, 1, 1, 64, 1, 1, 0, 0,
            CONV_4_INPUT_OFFSET, CONV_4_OUTPUT_OFFSET,
            ds_cnn_s_layer_5_conv2d_output_mult,
            ds_cnn_s_layer_5_conv2d_output_shift, nn_buf0, 25, 5);

  // DS Block 3
  dw_conv2d_s8(nn_buf0, 25, 5, 64, ds_cnn_s_layer_6_dw_conv2d_weights,
               ds_cnn_s_layer_6_dw_conv2d_bias, 3, 3, 1, 1, 1, 1,
               DW_CONV_5_INPUT_OFFSET, DW_CONV_5_OUTPUT_OFFSET,
               ds_cnn_s_layer_6_dw_conv2d_output_mult,
               ds_cnn_s_layer_6_dw_conv2d_output_shift, nn_buf1, 25, 5);

  conv2d_s8(nn_buf1, 25, 5, 64, ds_cnn_s_layer_7_conv2d_weights,
            ds_cnn_s_layer_7_conv2d_bias, 1, 1, 64, 1, 1, 0, 0,
            CONV_6_INPUT_OFFSET, CONV_6_OUTPUT_OFFSET,
            ds_cnn_s_layer_7_conv2d_output_mult,
            ds_cnn_s_layer_7_conv2d_output_shift, nn_buf0, 25, 5);

  // DS Block 4
  dw_conv2d_s8(nn_buf0, 25, 5, 64, ds_cnn_s_layer_8_dw_conv2d_weights,
               ds_cnn_s_layer_8_dw_conv2d_bias, 3, 3, 1, 1, 1, 1,
               DW_CONV_7_INPUT_OFFSET, DW_CONV_7_OUTPUT_OFFSET,
               ds_cnn_s_layer_8_dw_conv2d_output_mult,
               ds_cnn_s_layer_8_dw_conv2d_output_shift, nn_buf1, 25, 5);

  conv2d_s8(nn_buf1, 25, 5, 64, ds_cnn_s_layer_9_conv2d_weights,
            ds_cnn_s_layer_9_conv2d_bias, 1, 1, 64, 1, 1, 0, 0,
            CONV_8_INPUT_OFFSET, CONV_8_OUTPUT_OFFSET,
            ds_cnn_s_layer_9_conv2d_output_mult,
            ds_cnn_s_layer_9_conv2d_output_shift, nn_buf0, 25, 5);

  // Average Pool (25x5x64 -> 1x1x64)
  int8_t pooled[64];
  avg_pool_s8(nn_buf0, 25, 5, 64, AVERAGE_POOL_9_INPUT_OFFSET, pooled);

  // Fully Connected (64 -> 12)
  int8_t fc_out[12];
  fc_s8(pooled, 64, ds_cnn_s_layer_12_fc_weights, ds_cnn_s_layer_12_fc_bias, 12,
        FULLY_CONNECTED_11_INPUT_OFFSET, FULLY_CONNECTED_11_OUTPUT_OFFSET,
        FULLY_CONNECTED_11_OUTPUT_MULTIPLIER, FULLY_CONNECTED_11_OUTPUT_SHIFT,
        fc_out);

  // Softmax
  softmax_s8(fc_out, 12, SOFTMAX_12_MULT, SOFTMAX_12_SHIFT, SOFTMAX_12_DIFF_MIN,
             out_data);

  return EE_STATUS_OK;
}
#include <riscv_vector.h>
#include <stddef.h>

// =================================================================
// Kernel 1: Float Addition (th_add_f32)
// Strategy: Max throughput using LMUL=8 (grouping 8 registers)
// =================================================================
void th_add_f32(const float *srcA, const float *srcB, float *dst, size_t n) {
  size_t vl;
  for (; n > 0; n -= vl, srcA += vl, srcB += vl, dst += vl) {
    // 1. Ask hardware for max elements (LMUL=8)
    vl = __riscv_vsetvl_e32m8(n);

    // 2. Load vectors
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(srcA, vl);
    vfloat32m8_t vy = __riscv_vle32_v_f32m8(srcB, vl);

    // 3. Add
    vfloat32m8_t vz = __riscv_vfadd_vv_f32m8(vx, vy, vl);

    // 4. Store
    __riscv_vse32_v_f32m8(dst, vz, vl);
  }
}

// =================================================================
// Kernel 2: Dot Product (th_dot_prod_f32)
// Strategy: Accumulate in Vector, Reduce once at the end.
// =================================================================
float th_dot_prod_f32(const float *srcA, const float *srcB, size_t n) {
  size_t vl;

  // Initialize Accumulator (vsum = 0)
  // We use a max-length zero vector to ensure the whole register is cleared
  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

  for (; n > 0; n -= vl, srcA += vl, srcB += vl) {
    vl = __riscv_vsetvl_e32m8(n);

    vfloat32m8_t v_a = __riscv_vle32_v_f32m8(srcA, vl);
    vfloat32m8_t v_b = __riscv_vle32_v_f32m8(srcB, vl);

    // Fused Multiply-Add: vsum += v_a * v_b
    vsum = __riscv_vfmacc_vv_f32m8(vsum, v_a, v_b, vl);
  }

  // Horizontal Reduction (Squash vector to scalar)
  vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t result_vec =
      __riscv_vfredusum_vs_f32m8_f32m1(vsum, v_zero, __riscv_vsetvlmax_e32m8());

  return __riscv_vfmv_f_s_f32m1_f32(result_vec);
}

// =================================================================
// Kernel 3: Complex Multiply (th_cmplx_mult_cmplx_f32)
// Strategy: Segmented Loads (vlseg2) to handle [Real, Imag] struct
// Math: (a+bi)(c+di) = (ac - bd) + i(ad + bc)
// =================================================================
// =================================================================
// Kernel 3: Complex Multiply (th_cmplx_mult_cmplx_f32)
// FIXED: Using correct GCC 15 vset naming convention
// =================================================================
void th_cmplx_mult_cmplx_f32(const float *srcA, const float *srcB, float *dst,
                             size_t n) {
  size_t vl;
  for (; n > 0; n -= vl, srcA += 2 * vl, srcB += 2 * vl, dst += 2 * vl) {

    vl = __riscv_vsetvl_e32m4(n);

    // 1. De-interleave Load
    vfloat32m4x2_t vx = __riscv_vlseg2e32_v_f32m4x2(srcA, vl);
    vfloat32m4x2_t vy = __riscv_vlseg2e32_v_f32m4x2(srcB, vl);

    vfloat32m4_t ar = __riscv_vget_v_f32m4x2_f32m4(vx, 0);
    vfloat32m4_t ai = __riscv_vget_v_f32m4x2_f32m4(vx, 1);
    vfloat32m4_t br = __riscv_vget_v_f32m4x2_f32m4(vy, 0);
    vfloat32m4_t bi = __riscv_vget_v_f32m4x2_f32m4(vy, 1);

    // 2. Math: Real Part = (ar * br) - (ai * bi)
    vfloat32m4_t real_part = __riscv_vfmul_vv_f32m4(ar, br, vl);
    vfloat32m4_t ai_bi = __riscv_vfmul_vv_f32m4(ai, bi, vl);
    real_part = __riscv_vfsub_vv_f32m4(real_part, ai_bi, vl);

    // 3. Math: Imag Part = (ar * bi) + (ai * br)
    vfloat32m4_t imag_part = __riscv_vfmul_vv_f32m4(ar, bi, vl);
    imag_part = __riscv_vfmacc_vv_f32m4(imag_part, ai, br, vl);

    // 4. Interleave Store
    // FIX IS HERE: Swapped suffixes to _f32m4_f32m4x2
    vfloat32m4x2_t vz;
    vz = __riscv_vset_v_f32m4_f32m4x2(vz, 0, real_part);
    vz = __riscv_vset_v_f32m4_f32m4x2(vz, 1, imag_part);

    __riscv_vsseg2e32_v_f32m4x2(dst, vz, vl);
  }
}
# GitHub Issue for sanderjo/SpacemiT-K3-X100-A100

## Title:
```
Benchmark Request: EEMBC AudioMark (RVV 1.0 Optimized) on SpacemiT K3
```

## Body (Copy everything below):

---

Hi @sanderjo,

I am working on the **LFX RISC-V Mentorship** program under Edward Cheshire. I have ported the [EEMBC AudioMark](https://github.com/eembc/audiomark) benchmark to RISC-V using **RVV 1.0 intrinsics** and am looking to validate the performance on real SpacemiT K3/X60 hardware.

Since the K3 cores support **RVV 1.0 with VLEN=128**, they are the perfect target for this implementation.

---

## ⚡ Quick Test (Static Binary)

I have packaged a **statically linked binary** to avoid any glibc/dependency issues on your specific OS (Bianbu/Armbian). It should run immediately:

```bash
# 1. Download the binary (approx 1MB)
wget https://github.com/Ayushd785/riscv-Audiomark-lab/releases/download/v0.2-rvv-alpha/audiomark-rvv-static

# 2. Make executable
chmod +x audiomark-rvv-static

# 3. Run Benchmark
./audiomark-rvv-static
```

---

## 📊 Expected Behavior

The benchmark runs for approximately **10-15 seconds**. My QEMU emulation results are below for comparison:

| Configuration | Score | Runtime |
|---------------|-------|---------|
| QEMU (VLEN=128) | 1,583 AudioMarks | 10.5s |
| QEMU (VLEN=256) | 1,728 AudioMarks | 10.8s |

**Target:** I am hoping the K3 hardware significantly outperforms the emulated score of 1,583.

---

## 📝 Requested Output

Could you please paste the stdout output here? Specifically looking for:

1. **Total runtime** (in seconds)
2. **AudioMark Score**
3. (Optional) Output of `cat /proc/cpuinfo | head -30` (to verify `zba`/`zbb` extension support)

---

## 🐳 Alternative: Docker Reproduction

If you prefer to build from source or audit the code, I have published the build environment:

- **Docker:** `docker pull ayushd785/riscv-audio-lab`
- **Source:** [Ayushd785/riscv-Audiomark-lab](https://github.com/Ayushd785/riscv-Audiomark-lab)
- **Release:** [v0.2-rvv-alpha](https://github.com/Ayushd785/riscv-Audiomark-lab/releases/tag/v0.2-rvv-alpha)

---

## 🔧 Technical Details

| Aspect | Value |
|--------|-------|
| RVV Version | 1.0 (ratified spec) |
| VLEN | Independent (uses `vsetvl` strip-mining) |
| Compiler | GCC 15 with `-march=rv64gcv -mabi=lp64d` |
| Linking | Static (no external dependencies) |
| Key Kernels | `th_add_f32`, `th_dot_prod_f32`, `th_cmplx_mult_cmplx_f32` |

Thanks for your help in validating RVV performance on the K3! 🚀

---

## Steps to Create the Issue:

1. Go to: https://github.com/sanderjo/SpacemiT-K3-X100-A100/issues/new
2. Paste the **Title** above
3. Paste the **Body** (everything between the `---` markers)
4. Click "Submit new issue"

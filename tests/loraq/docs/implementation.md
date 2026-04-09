# fast_loraq — Implementation Notes

## Overview

`fast_loraq` provides Triton-accelerated linear layers for quantized inference
on AMD MI350 GPUs.  Five layer classes are available, each with different
precision/speed trade-offs:

| Layer | Weight format | Activation | Output | Rank | Phase 2 |
|---|---|---|---|---|---|
| `TritonLinear` | fp16/bf16 | fp16/bf16 | fp16/bf16 | — | — |
| `TritonLinearFP4` | MXFP4 e2m1 | online FP4 quant | bf16 | — | — |
| `TritonLinearLoRA` | MXFP4 + fp16 L,R | fp16 | bf16 | 32 | — |
| `TritonLinearLoRaQ` | MXFP4 + MXFP8 L,R | pre-quant FP8 | MXFP8 | 64 | tl.dot fp16 |
| `TritonLinearLoRaQFP8` | MXFP4 + MXFP8 L,R | pre-quant FP8 | MXFP8 | 64 | dot_scaled fp8 |

All tensors (weights, biases, LoRA factors) are stored as **non-trainable
buffers** (`register_buffer`).  No `nn.Parameter` is used — these layers are
purely for inference.

---

## Project structure

```
fast_loraq/
├── fast_loraq/
│   ├── __init__.py              # Public API (re-exports all layers + quant utils)
│   ├── kernels.py               # 8 Triton JIT kernels
│   ├── linear.py                # 5 nn.Module layer classes + Python wrappers
│   ├── quant.py                 # MXFP4/MXFP8 quantization + dequantization
│   └── autotune_configs.py      # Autotune configs + AutotunedLoRaQ runtime tuner
├── tests/
│   ├── __init__.py
│   └── test_linear.py           # Correctness tests (pytest)
├── benchmarks/
│   ├── bench_linear.py          # TFLOPS benchmark (all layers)
│   ├── sweep_loraq.py           # Hyperparameter sweep for LoRaQ kernels
│   └── profile_loraq.py         # Proton profiling + assembly inspection
├── docs/
│   └── implementation.md        # This document
└── pyproject.toml
```

---

## Kernels (`kernels.py`)

Eight Triton JIT kernels:

| # | Kernel | Purpose |
|---|---|---|
| 1 | `matmul_kernel` | Basic fp16/bf16 tiled GEMM |
| 2 | `matmul_fp4_kernel` | MXFP4 GEMM via `tl.dot_scaled("e2m1","e2m1")` |
| 3 | `_mxfp4_quant_kernel` | fp16/bf16 → packed e2m1 + e8m0 quantiser |
| 4 | `_mxfp8_quant_kernel` | fp16/bf16 → float8_e4m3fn + e8m0 quantiser |
| 5 | `loraq_project_and_quant_kernel` | Fused A@R^T projection + MXFP4 quant of A |
| 6 | `loraq_dual_gemm_kernel` | Fused P@L^T + Q(A)@Q(W)^T dual GEMM |
| 7 | `loraq_fused_q8_kernel` | LoRaQ fused FP8/FP4, Phase 2 = `tl.dot` fp16 |
| 8 | `loraq_fused_q8_scaled_kernel` | LoRaQ fused FP8/FP4, Phase 2 = `dot_scaled` fp8 |

### Kernels 7 & 8 — LoRaQ fused kernels

Both compute the same formula in a single kernel launch:

```
C_fp8 = MXFP8_quant( P × L^T  +  A_fp8 × W_fp4^T  [+ bias] )
```

where `P = A_fp8 × R_fp8^T` is computed inline.

**Three phases per output tile:**

| Phase | Operation | Kernel 7 (V1) | Kernel 8 (V2) |
|---|---|---|---|
| 1 | A×R^T + A×W^T | `dot_scaled("e4m3","e4m3")` + `dot_scaled("e4m3","e2m1")` | Same |
| 2 | P × L^T | P→fp16, L dequant fp8→fp16, `tl.dot` | P→fp8 in-register, `dot_scaled("e4m3","e4m3")` |
| 3 | Sum + quant | In-register MXFP8 quantization (3D reshape for group-wise scale) | Same |

**V1 vs V2 trade-off:** V1 keeps P in fp16 (higher fidelity); V2 re-quantizes P to fp8
(more noise but potentially higher throughput from hardware-accelerated scaled-dot).

---

## Layer classes (`linear.py`)

### `TritonLinear`

Basic fp16/bf16 linear layer using Kernel 1.

```python
layer = TritonLinear(4096, 4096, bias=True, dtype=torch.float16)
y = layer(x)  # x: (..., 4096) → y: (..., 4096)
```

### `TritonLinearFP4`

MXFP4 weight, online activation quantization, Kernel 2.

```python
layer = TritonLinearFP4.from_float(nn.Linear(4096, 4096))
y = layer(x)  # x: (..., 4096) fp16 → y: (..., 4096) bf16
```

### `TritonLinearLoRA`

LoRA + MXFP4 with fp16 L,R factors (rank=32), Kernels 5+6.

```python
layer = TritonLinearLoRA.from_float(nn.Linear(4096, 4096))
y = layer(x)  # x: (..., 4096) fp16 → y: (..., 4096) bf16
```

### `TritonLinearLoRaQ`

Fused FP8/FP4 LoRaQ with MXFP8 I/O (rank=64), Kernel 7.
Activation must be pre-quantized to MXFP8.

```python
layer = TritonLinearLoRaQ.from_float(nn.Linear(4096, 4096))
a_fp8, a_scale = dynamic_mxfp8_quant(x)
c_fp8, c_scale = layer(a_fp8, a_scale)
# Dequant for inspection: y = mxfp8_to_f32(c_fp8, c_scale)
```

### `TritonLinearLoRaQFP8`

Same as LoRaQ but Phase 2 uses `dot_scaled` fp8 (Kernel 8).
Inherits from `TritonLinearLoRaQ` — same buffers, factories, interface.

```python
layer = TritonLinearLoRaQFP8.from_float(nn.Linear(4096, 4096))
c_fp8, c_scale = layer(a_fp8, a_scale)
```

### Factory methods

All quantized layers provide:

- **`from_float(nn.Linear)`** — quantizes weight (+ SVD for LoRA factors)
- **`from_weight(tensor, bias=None)`** — quantizes from raw weight tensor

### Buffer-only design

All tensors use `register_buffer` (no `nn.Parameter`).  This means:
- No gradients tracked
- Correct `model.to(device)` and `state_dict()` behavior
- `model.parameters()` returns empty — inference only

---

## Quantization utilities (`quant.py`)

| Function | Input | Output |
|---|---|---|
| `dynamic_mxfp4_quant(x)` | (M,N) fp16 | `(M,N//2)` uint8 packed + `(M,N//32)` uint8 scales |
| `dynamic_mxfp8_quant(x)` | (M,N) fp16 | `(M,N)` float8_e4m3fn + `(M,N//32)` uint8 scales |
| `mxfp4_to_f32(x_fp4)` | packed uint8 | fp32 (via LUT) |
| `mxfp8_to_f32(x_fp8, scales)` | fp8 + scales | fp32 |
| `e8m0_to_f32(scales)` | uint8 e8m0 | fp32 power-of-two multipliers |

---

## Autotuning (`autotune_configs.py`)

### Config definitions

`LORAQ_Q8_CONFIGS` contains 15 `triton.Config` objects sweeping:

| Parameter | Values |
|---|---|
| `BLOCK_M` | 64, 128, 256 |
| `BLOCK_N` | 64, 128, 256 |
| `BLOCK_K` | 64, 128 |
| `GROUP_SIZE_M` | 1, 4, 8 |
| `num_warps` | 4, 8 |
| `num_stages` | 1, 2 |

### Runtime autotuner

`AutotunedLoRaQ` wraps any LoRaQ kernel with runtime autotuning:

```python
from fast_loraq.autotune_configs import AutotunedLoRaQ
from fast_loraq.kernels import loraq_fused_q8_kernel

autotuned = AutotunedLoRaQ(loraq_fused_q8_kernel)
# First call benchmarks all 15 configs, caches the best:
c_fp8, c_scale = autotuned(a_fp8, a_scale, r_fp8, r_scale,
                            l_fp8, l_scale, w_fp4_t, w_scale,
                            M, N, K)
# Subsequent calls with same (M,N,K) use cached best config instantly.
```

---

## Running the code

### Installation

```bash
cd fast_loraq
pip install -e ".[dev]"
```

### Tests

```bash
# Run all tests
pytest tests/ -v

# Run only LoRaQ FP8 tests
pytest tests/test_linear.py -v -k "loraq_q8 or loraq_fp8 or loraq_v1_vs_v2"

# Run only dimension/rank rejection tests (no GPU needed)
pytest tests/test_linear.py -v -k "rejects"
```

### Benchmarks

```bash
# Full benchmark suite (all layers)
python -m benchmarks.bench_linear

# fp16/bf16 only
python -m benchmarks.bench_linear --fp-only

# MXFP4 only
python -m benchmarks.bench_linear --fp4-only

# LoRA+Q (rank-32, fp16 L/R)
python -m benchmarks.bench_linear --loraq-only

# LoRaQ FP8: V1 vs V2 vs FP4 vs fp16
python -m benchmarks.bench_linear --loraq-q8-only

# Autotuned vs fixed-config comparison
python -m benchmarks.bench_linear --autotuned-only

# Export results to JSON
python -m benchmarks.bench_linear --loraq-q8-only --json results.json
```

### Hyperparameter sweep

Find the optimal `(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)`
for each (M, K, N) problem size:

```bash
# Full sweep: V1 and V2, all 216 config combinations per size
python -m benchmarks.sweep_loraq

# Quick sweep: max 10 configs per size
python -m benchmarks.sweep_loraq --max-configs 10

# Sweep only V1 (tl.dot fp16 Phase 2)
python -m benchmarks.sweep_loraq --v1-only

# Sweep only V2 (dot_scaled fp8 Phase 2)
python -m benchmarks.sweep_loraq --v2-only

# Sweep + verify MFMA scaled-dot instructions in compiled assembly
python -m benchmarks.sweep_loraq --check-asm

# Export sweep results to JSON
python -m benchmarks.sweep_loraq --json sweep_results.json
```

### Profiling with Proton

Profile kernels with Triton's Proton profiler and generate trace files:

```bash
# Profile default size (128 × 4096 × 4096)
python -m benchmarks.profile_loraq

# Profile specific size
python -m benchmarks.profile_loraq --size 256 4096 4096

# Profile all standard sizes
python -m benchmarks.profile_loraq --all-sizes

# Profile + assembly inspection
python -m benchmarks.profile_loraq --check-asm

# Custom output directory for traces
python -m benchmarks.profile_loraq --output my_traces/

# Export timing results to JSON
python -m benchmarks.profile_loraq --all-sizes --json profile_results.json
```

Trace files are saved to the `--output` directory (default: `traces/`).
View them with:
- **Chrome:** `chrome://tracing` → Load
- **Perfetto:** https://ui.perfetto.dev/

### Assembly inspection

The `--check-asm` flag scans the Triton compilation cache for compiled
AMDGCN assembly and counts occurrences of key instructions:

| Instruction | Meaning |
|---|---|
| `v_mfma_scale_f32_32x32x64_f8f6f4` | Hardware-accelerated scaled MFMA (32×32 tile) — confirms `dot_scaled` is using MI350 hardware |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | Alternative scaled MFMA (16×16 tile) |
| `v_mfma_f32_32x32x16_f16` | Standard fp16 MFMA — used by `tl.dot` in V1 Phase 2 |
| `v_mfma_f32_16x16x32_f16` | Alternative fp16 MFMA tile |

**Expected results:**
- Both V1 and V2 should show `v_mfma_scale_f32` for Phase 1 (`A×R^T` and `A×W^T`)
- V1 should also show `v_mfma_f32_*_f16` for Phase 2 (`P×L^T` in fp16)
- V2 should show only `v_mfma_scale_f32` (Phase 2 also uses scaled dot)

---

## LoRaQ architecture details

### Formula

```
C = fp16(q8(A) @ q8(R)^T) @ dequant_fp16(q8(L))^T  +  q8(A) @ q4(W)^T  [+ bias]
```

(V2 variant: P is re-quantized to fp8 instead of cast to fp16)

### Dimensions

| Matrix | Shape | Format | Storage |
|---|---|---|---|
| A (activation) | (M, K) | float8_e4m3fn | Pre-quantized input |
| A_scale | (M, K//32) | uint8 e8m0 | Pre-quantized input |
| W (weight) | (N, K//2) | uint8 packed e2m1 | Buffer |
| W_scale | (N, K//32) | uint8 e8m0 | Buffer |
| R (low-rank) | (64, K) | float8_e4m3fn | Buffer |
| R_scale | (64, K//32) | uint8 e8m0 | Buffer |
| L (low-rank) | (N, 64) | float8_e4m3fn | Buffer |
| L_scale | (N, 2) | uint8 e8m0 | Buffer (2 groups of 32 in rank dim) |
| bias | (N,) | float32 | Buffer (optional) |
| C (output) | (M, N) | float8_e4m3fn | Computed |
| C_scale | (M, N//32) | uint8 e8m0 | Computed |

### Kernel tile sizes

Default config: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_SIZE_M=8, num_warps=8, num_stages=2`

Constraints:
- `BLOCK_K ≥ 64` (minimum for `dot_scaled` with e2m1 format)
- `BLOCK_K` multiple of 32 (scale group size)
- `in_features` must be divisible by 64
- `out_features` must be divisible by 32
- `rank` must be 64

### Memory footprint (per layer)

```
FP4 weight:  N × K/2 + N × K/32  bytes  (packed e2m1 + e8m0 scales)
R factor:    64 × K + 64 × K/32  bytes  (fp8 + scales)
L factor:    N × 64 + N × 2      bytes  (fp8 + scales)
Total ≈ N×K×(17/32) + 66K + 66N  bytes
```

For K=N=4096: ~9.3 MB (vs 32 MB for fp16 — **71% reduction**).

---

---

## Session log — adaptive tile config for `TritonLinear` (2026-03-31)

### Problem

`TritonLinear` was using fixed tile parameters
(`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3`) for all
problem sizes.  At small M (e.g. M=128, K=N=4096) this creates only
ceil(128/128) × ceil(4096/128) = **32 output tiles**, leaving ~90% of the
304 CUs idle:

```
M=128, old fixed BM=128:  1 M-tile × 32 N-tiles = 32 tiles  → 32/304 CUs busy
M=128, hipBLAS MT32:      4 M-tiles × 32 N-tiles = 128 tiles → 42% CU utilization
```

hipBLAS was measured at ~30µs for M=128; our fixed kernel was at ~45µs (~1.5×
slower) for the same reason.

### Fix — `_pick_tile_config(M, N)` adaptive table

Replaced the fixed constants with a per-M config table in `linear.py`:

```python
_TILE_CONFIGS = [
    # (max_M, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, nonkdim)
    (32,   16,  64,  64, 4, 2, 16),   # M ≤  32
    (64,   32,  64,  64, 4, 2, 32),   # M ≤  64
    (128,  32, 128, 128, 4, 2, 32),   # M ≤ 128  ← key fix: BM=32 → 128 tiles
    (256,  64, 128, 128, 8, 3, 32),   # M ≤ 256
    (None, 128, 128, 128, 8, 2, 32),  # M > 256
]
```

Both `triton_matmul` and `triton_matmul_nt` were updated to drive all kernel
launch parameters (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`,
`matrix_instr_nonkdim`) from this table instead of hard-coded literals.

**Effect of the M≤128 fix:**

| | Before | After |
|---|---|---|
| BLOCK_M | 128 | **32** |
| Output tiles (M=128, N=4096) | 32 | **128** |
| Triton TFLOPS | 78.3 | **98.7** |

This matches hipBLAS's reported `MT32×128×128` selection for M=128.

### What we tried and why it failed

#### Attempt 1 — widen BLOCK_N to 256 for large M

**Hypothesis:** BLOCK_N=256 doubles B-matrix L2 reuse and halves the grid
launch overhead.

**Config tried:** `(None, 128, 256, 128, 8, 4, 32)` — M>256

**What happened:** Triton raised `OutOfResources: shared memory Required=294912,
Hardware limit=163840`. LDS budget:

```
A tile: 128 × 128 × 2 bytes = 32 KB
B tile: 256 × 128 × 2 bytes = 64 KB
Per stage: 96 KB
× 4 stages = 384 KB  >> 160 KB hardware limit
```

**Root cause:** BN=256 × BK=128 produces a 64KB B tile. Even with num_stages=1
the A+B tiles require 96 KB, and Triton's pipeline allocates num_stages copies.

---

#### Attempt 2 — BLOCK_N=256 + BLOCK_K=64 to stay within LDS

**Hypothesis:** Halving BK from 128→64 halves the tile area and lets 3-stage
pipelining fit.

**Config tried:** `(None, 128, 256, 64, 8, 3, 32)` — M>256

**LDS:** 3 × (128×64 + 256×64) × 2 = 147 456 bytes — fits within 160 KB ✓

**What happened:**

| Size | Triton TFLOPS | Speedup vs Torch |
|---|---|---|
| M=128 | 98.1 | 0.84× |
| M=1024 | **352** | 0.45× |

M=1024 regressed catastrophically from ~560 TFLOPS to 352 TFLOPS.

**Root cause — CU starvation:**

```
BM=128, BN=256, M=1024, N=4096:
  grid = ceil(1024/128) × ceil(4096/256) = 8 × 16 = 128 tiles
  128 tiles on 304 CUs → only 42% CU utilization
```

BN=256 halves the N-tile count, leaving more than half the CUs idle.
Additionally, BK=64 doubles the K-loop iterations (64 → 64 iterations for
K=4096) and reduces MFMA saturation per K-step.

---

#### Attempt 3 — BLOCK_M=64 to increase tile count and reduce VGPR pressure

**Hypothesis:** BM=64 halves VGPR pressure vs BM=128, allowing 2 CTAs/CU
(512 threads/CU instead of 256), and produces 512 tiles at M=1024 vs 256.

**Config tried:** `(None, 64, 128, 128, 8, 3, 32)` — M>128

**LDS:** 3 × (64×128 + 128×128) × 2 = 147 456 bytes — fits ✓

**What happened:**

| Size | Triton TFLOPS | Speedup vs Torch |
|---|---|---|
| M=128 | 98.1 | 0.84× |
| M=1024 | **365** | 0.52× |

Still worse than the baseline 511–560 TFLOPS at M=1024.

**Root cause:** While 512 tiles provides higher theoretical CU occupancy, the
profiler showed that M=1024 / N=4096 with BM=64 launches 2 full waves on the
304-CU GPU. The second wave has no way to hide its latency once the first wave
drains — no persistent scheduling. The extra K-loop overhead (32 iterations for
BK=128 is already near-optimal for fp16 MFMA) and the reduced per-tile compute
leave MFMA underutilized.

---

### Final config chosen

```python
(None, 128, 128, 128, 8, 2, 32),  # M > 256
```

- **BM=128, BN=128**: 256 tiles at M=1024/N=4096 (84% CU utilization for 1 wave)
- **BK=128**: 32 K-iterations — maximum MFMA saturation per loop
- **num_stages=2**: double buffering — LDS = 2 × (128+128) × 128 × 2 = 128 KB < 160 KB

### Observed performance (stable across runs)

| Size | Triton TFLOPS | Torch TFLOPS | Speedup |
|---|---|---|---|
| M=128, K=N=4096 | 98.7 | 116.6 | 0.85× |
| M=1024, K=N=4096 | 511 | 603 | 0.85× |

**Note on Torch timing variance:** Torch's timing swings ±49% between
identical runs on this GPU (observed range: 36.7–54.7µs for M=128). Triton
timing is stable (43–45µs). The "1.21× speedup" observed in an earlier
session was the result of a high Torch measurement (54.7µs), not a Triton
improvement. The actual Triton TFLOPS was 94.4 then vs 98.7 now — Triton is
slightly faster, but the Torch reference is lower in recent runs.

### Remaining gap to hipBLAS

hipBLAS achieves ~676 TFLOPS at M=1024/K=N=4096. Closing this gap requires:

1. **Autotuning** (`triton.autotune`) — sweep over BM, BN, BK, warps, stages
   for each (M, K, N) size at runtime and cache the best config
2. **Persistent / stream-K scheduling** — eliminates tail-wave inefficiency by
   having CUs pull tiles from a global counter rather than using a static grid
3. **Assembly-level optimisation** — hipBLAS uses hand-tuned AMDGCN assembly
   for the innermost MFMA loop; Triton's compiler leaves ~15–20% of peak MFMA
   throughput on the table

These are outside the scope of the tile-config table and would require
significant additional work.

---

## Test summary

| Test category | Test count | Requires GPU |
|---|---|---|
| fp16/bf16 matmul correctness | 10 | ✓ |
| TritonLinear vs nn.Linear | 12 | ✓ |
| MXFP4 quantization | 4 | ✓ |
| TritonLinearFP4 | 6 | ✓ |
| TritonLinearLoRA | 9 | ✓ / ✗ |
| MXFP8 quantization | 1 | ✓ |
| TritonLinearLoRaQ (V1) | 8 | ✓ / ✗ |
| TritonLinearLoRaQFP8 (V2) | 5 | ✓ |
| V1 vs V2 comparison | 1 | ✓ |
| Dimension/rank rejection | 4 | ✗ |

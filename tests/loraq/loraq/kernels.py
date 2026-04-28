"""
Triton matrix multiplication kernels for fast_loraq.

Eleven kernels:
  1. matmul_kernel                       -- basic fp16/bf16 tiled GEMM
  2. matmul_fp4_kernel                   -- MXFP4 (e2m1) GEMM using tl.dot_scaled
  3. _mxfp4_quant_kernel                 -- fp16/bf16 -> packed e2m1 + e8m0 quantiser
  4. _mxfp8_quant_kernel                 -- fp16/bf16 -> float8_e4m3fn + e8m0 quantiser
  5. loraq_project_and_quant_kernel      -- fused A @ R^T projection + MXFP4 quant of A
  6. loraq_dual_gemm_kernel              -- fused P @ L^T + Q(A) @ Q(W)^T dual GEMM
  7. loraq_fused_q8_kernel               -- fused FP8/FP4 LoRA+Q, Phase 2 = tl.dot fp16
  8. loraq_fused_q8_scaled_kernel        -- fused FP8/FP4 LoRA+Q, Phase 2 = dot_scaled fp8
  9. loraq_fused_fp16io_kernel           -- LoRaQ.3: fp16 in (fused quant), fp16 out
 10. loraq_split_proj_kernel             -- LoRaQ.4 K_proj: A_fp8 @ R_fp8^T → P_fp16
 11. loraq_split_main_kernel             -- LoRaQ.4 K_main: P×L^T + A×W^T → C_fp8
 12. loraq_fused_q8_perkiter_kernel      -- LoRaQ.5: per-K-iter correction (no acc_p)
 13. loraq_fused_q8_fp16lr_kernel        -- LoRaQ.6: like K7 but L,R are fp16
"""

import triton
import triton.language as tl


# ===========================================================================
# Kernel 1 -- fp16 / bf16 GEMM
# ===========================================================================

@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Compute C = A @ B where:
        A is (M, K), B is (K, N), C is (M, N)

    Each program instance computes a (BLOCK_M, BLOCK_N) tile of C.
    B may be passed with transposed strides to compute A @ B^T without copying.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Swizzle: group programs for better L2 cache reuse.
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the tile this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first block of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Row/col masks that are loop-invariant — computed once
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: iterate over the K dimension in BLOCK_K steps
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            # No boundary check needed along K — compiler can eliminate predicate
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_offs = k * BLOCK_K + offs_k
            a = tl.load(a_ptrs, mask=m_mask[:, None] & (k_offs[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C_ptr.type.element_ty)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


# ===========================================================================
# Kernel 2 -- MXFP4 (e2m1) GEMM with block-scaled dot_scaled
# ===========================================================================

@triton.jit
def matmul_fp4_kernel(
    # Pointers to packed FP4 matrices and their e8m0 scales
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
    # Matrix dimensions (logical, unpacked)
    M, N, K,
    # A strides  -- A is (M, K//2) packed uint8
    stride_am, stride_ak,
    # B strides  -- B is (K//2, N) packed uint8
    stride_bk, stride_bn,
    # C strides  -- C is (M, N) in fp16/bf16
    stride_cm, stride_cn,
    # A_scale strides -- (M, K//32) uint8 e8m0
    stride_asm, stride_ask,
    # B_scale strides -- (N, K//32) uint8 e8m0
    stride_bsn, stride_bsk,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = dequant(A, A_scale) @ dequant(B, B_scale)^T

    A and B are in MXFP4 e2m1 format: two 4-bit values packed per uint8.
    Every 32 elements along K share one e8m0 scale (a pure exponent).

    The heavy lifting is done by ``tl.dot_scaled`` which fuses the
    dequantisation and dot product into a single hardware instruction
    on AMD MI350-class GPUs.

    Dimensions
    ----------
    A      : (M, K//2)   packed uint8
    B      : (K//2, N)   packed uint8
    A_scale: (M, K//32)  uint8 (e8m0)
    B_scale: (N, K//32)  uint8 (e8m0)  -- note: row-major over N
    C      : (M, N)      fp16 or bf16
    """
    # Every 32 logical elements share one scale
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly swizzle (same grouping as fp16 kernel)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row / column indices for this tile (with modulo wrap for safety)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # K-offsets in *packed* space (2 values per byte)
    rk = tl.arange(0, BLOCK_K // 2)
    # K-offsets for scale pointers (one scale per 32 elements)
    rks = tl.arange(0, BLOCK_K // SCALE_GROUP)

    # Base pointers for A, B data tiles
    A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # Base pointers for scale tiles
    A_scale_BASE = A_scale_ptr + rm[:, None] * stride_asm + rks[None, :] * stride_ask
    # B_scale is (N, K//32) even though B data is (K//2, N)
    B_scale_BASE = B_scale_ptr + rn[:, None] * stride_bsn + rks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        # Load scales
        a_scales = tl.load(A_scale_BASE)
        b_scales = tl.load(B_scale_BASE)

        # Load packed FP4 data with boundary masking
        k_remaining = K // 2 - k * (BLOCK_K // 2)
        a = tl.load(A_BASE, mask=rk[None, :] < k_remaining, other=0)
        b = tl.load(B_BASE, mask=rk[:, None] < k_remaining, other=0)

        # Fused dequant + dot product (hardware-accelerated on MI350)
        acc = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", acc=acc, out_dtype=tl.float32)

        # Advance pointers along K
        A_BASE += (BLOCK_K // 2) * stride_ak
        B_BASE += (BLOCK_K // 2) * stride_bk
        A_scale_BASE += (BLOCK_K // SCALE_GROUP) * stride_ask
        B_scale_BASE += (BLOCK_K // SCALE_GROUP) * stride_bsk

    # Store result with bounds masking
    c = acc.to(C_ptr.type.element_ty)
    rm_s = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (rm_s[:, None] < M) & (rn_s[None, :] < N)
    C_ptrs = C_ptr + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
    tl.store(C_ptrs, c, mask=c_mask)


# ===========================================================================
# Kernel 3 -- MXFP4 quantization (fp16/bf16 -> packed e2m1 + e8m0 scales)
# ===========================================================================

@triton.jit
def _mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    scale_ptr,
    stride_x_m, stride_x_n,
    stride_fp4_m, stride_fp4_n,
    stride_s_m, stride_s_n,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANT_GROUP: tl.constexpr,
):
    """
    Quantize a (M, N) matrix to MXFP4 e2m1 with e8m0 block scales.

    Each group of QUANT_GROUP=32 elements along the N axis gets one
    e8m0 scale computed from the max absolute value.

    Outputs:
        x_fp4 : (M, N//2) uint8  -- two packed e2m1 nibbles per byte
        scale : (M, N//32) uint8 -- one e8m0 exponent per group
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load a (BLOCK_SIZE, QUANT_GROUP) tile of input
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * QUANT_GROUP + tl.arange(0, QUANT_GROUP)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n,
        mask=mask,
    ).to(tl.float32)

    # --- compute e8m0 scale per row within this group ---
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    # Round amax up to nearest power of 2
    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)
    # Unbiased exponent: log2(amax) - 2 (max e2m1 value is 6.0 ~ 2^2 * 1.5)
    scale_exp = tl.log2(amax).floor() - 2
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    quant_scale = tl.exp2(-scale_exp)

    # Quantize
    qx = x * quant_scale

    # e8m0 = exponent + 127 bias
    scale_e8m0 = scale_exp.to(tl.uint8) + 127

    # --- convert fp32 -> e2m1 nibble ---
    qx_u = qx.to(tl.uint32, bitcast=True)
    s = qx_u & 0x80000000          # sign
    e = (qx_u >> 23) & 0xFF        # fp32 exponent
    m = qx_u & 0x7FFFFF            # fp32 mantissa

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Handle denormals in e2m1 range
    adj = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adj, m)
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign|exp|mantissa, round-nearest-up, saturate to 3-bit magnitude
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1 = ((s >> 28) | e2m1_tmp).to(tl.uint8)

    # Pack two nibbles into one byte: low | (high << 4)
    e2m1 = tl.reshape(e2m1, [BLOCK_SIZE, QUANT_GROUP // 2, 2])
    evens, odds = tl.split(e2m1)
    packed = evens | (odds << 4)

    # Store packed FP4
    out_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_n = pid_n * QUANT_GROUP // 2 + tl.arange(0, QUANT_GROUP // 2)
    out_mask = (out_m < M)[:, None] & (out_n < (N // 2))[None, :]
    tl.store(
        x_fp4_ptr + out_m[:, None] * stride_fp4_m + out_n[None, :] * stride_fp4_n,
        packed, mask=out_mask,
    )

    # Store e8m0 scale
    s_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    s_mask = (s_m < M)[:, None]
    tl.store(
        scale_ptr + s_m[:, None] * stride_s_m + pid_n * stride_s_n,
        scale_e8m0, mask=s_mask,
    )


# ===========================================================================
# Kernel 4 -- MXFP8 quantization (fp16/bf16 -> float8_e4m3fn + e8m0 scales)
# ===========================================================================

@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    x_fp8_ptr,
    scale_ptr,
    stride_x_m, stride_x_n,
    stride_fp8_m, stride_fp8_n,
    stride_s_m, stride_s_n,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANT_GROUP: tl.constexpr,
):
    """
    Quantize a (M, N) matrix to MXFP8 e4m3 with e8m0 block scales.

    Each group of QUANT_GROUP=32 elements along the N axis gets one
    e8m0 scale computed from the max absolute value.

    Outputs:
        x_fp8 : (M, N) float8_e4m3fn  -- one byte per element (no packing)
        scale : (M, N//32) uint8      -- one e8m0 exponent per group
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load a (BLOCK_SIZE, QUANT_GROUP) tile of input
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * QUANT_GROUP + tl.arange(0, QUANT_GROUP)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n,
        mask=mask,
    ).to(tl.float32)

    # --- compute e8m0 scale per row within this group ---
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    # Round amax up to nearest power of 2
    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)
    # Unbiased exponent: log2(amax) - 7  (max e4m3 value is 448.0 ~ 2^8 * 1.75)
    scale_exp = tl.log2(amax).floor() - 7
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    quant_scale = tl.exp2(-scale_exp)

    # Normalise into e4m3 range
    qx = x * quant_scale

    # Clamp to e4m3 representable range [-448, 448]
    qx = tl.clamp(qx, min=-448.0, max=448.0)

    # e8m0 = exponent + 127 bias
    scale_e8m0 = scale_exp.to(tl.uint8) + 127

    # --- convert fp32 -> float8_e4m3fn via bit manipulation ---
    qx_u = qx.to(tl.uint32, bitcast=True)
    s = qx_u & 0x80000000           # sign bit
    e = (qx_u >> 23) & 0xFF         # fp32 biased exponent
    m = qx_u & 0x7FFFFF             # fp32 mantissa (23 bits)

    E8_BIAS: tl.constexpr = 127
    E4_BIAS: tl.constexpr = 7

    # Handle subnormals in e4m3 range: exponents below (E8_BIAS - E4_BIAS)
    # need implicit-1 shifted into the mantissa
    adj = tl.core.sub(E8_BIAS - E4_BIAS, e, sanitize_overflow=False)
    subnormal_m = (0x800000 | m) >> (adj + 1)
    m = tl.where(e < (E8_BIAS - E4_BIAS), subnormal_m, m)
    e = tl.where(e < (E8_BIAS - E4_BIAS), 0, e - (E8_BIAS - E4_BIAS))

    # Round mantissa from 23 bits to 3 bits (round-nearest-even)
    # Keep top 3 bits of mantissa, use bit 19 for rounding
    round_bit = (m >> 19) & 1
    m3 = (m >> 20) + round_bit
    # Handle mantissa overflow (carry into exponent)
    e = e + (m3 >> 3)
    m3 = m3 & 0x7

    # Saturate exponent to 4-bit range [0, 15]; e4m3fn has no inf/nan
    e = tl.minimum(e, 15)
    # If exponent saturated to 15, clamp mantissa to max (0x6 for e4m3fn)
    m3 = tl.where(e >= 15, tl.minimum(m3, 0x6), m3)

    # Pack: [sign(1)][exp(4)][mantissa(3)]
    fp8_packed = ((s >> 24) | (e << 3) | m3).to(tl.uint8)

    # Store FP8 output (one byte per element, no packing)
    tl.store(
        x_fp8_ptr + offs_m[:, None] * stride_fp8_m + offs_n[None, :] * stride_fp8_n,
        fp8_packed, mask=mask,
    )

    # Store e8m0 scale
    s_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    s_mask = (s_m < M)[:, None]
    tl.store(
        scale_ptr + s_m[:, None] * stride_s_m + pid_n * stride_s_n,
        scale_e8m0, mask=s_mask,
    )


# ===========================================================================
# Kernel 5 -- LoRA+Q: fused projection A @ R^T  +  MXFP4 quantization of A
# ===========================================================================

@triton.jit
def loraq_project_and_quant_kernel(
    # Input matrices
    A_ptr,          # (M, K) fp16 activation
    R_ptr,          # (RANK, K) fp16 low-rank factor (stored row-major)
    # Channel-wise quantization scale
    channel_scale_ptr,  # (K,) fp16/bf16 — per-column scale applied before quant
    # Outputs
    P_ptr,          # (M, RANK) fp16 projection result  = A @ R^T
    A_fp4_ptr,      # (M, K // 2) uint8 packed e2m1    = Q(A) data
    A_scale_ptr,    # (M, K // 32) uint8 e8m0           = Q(A) scales
    # Dimensions (constexpr for full compile-time optimization)
    M: tl.constexpr,              # number of rows in A
    K: tl.constexpr,              # number of columns in A (== number of columns in R)
    # Strides for A (M, K)
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # Strides for R (RANK, K)
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    # Strides for P (M, RANK)
    stride_pm: tl.constexpr, stride_pr: tl.constexpr,
    # Strides for A_fp4 (M, K // 2)
    stride_fp4_m: tl.constexpr, stride_fp4_n: tl.constexpr,
    # Strides for A_scale (M, K // 32)
    stride_sm: tl.constexpr, stride_sn: tl.constexpr,
    # Stride for channel_scale (K,)
    stride_cs: tl.constexpr,
    # Compile-time constants
    RANK: tl.constexpr,         # fixed at 32
    BLOCK_M: tl.constexpr,      # rows per program
    QUANT_GROUP: tl.constexpr,  # 32 (== RANK)
):
    """
    Fused kernel that computes two results in a single pass over A:

    1. **Projection**: ``P = A @ R^T``  where R is ``(rank, K)`` and
       rank = QUANT_GROUP = 32 so the entire R row fits in one tile.
    2. **MXFP4 quantisation**: ``(A_fp4, A_scale) = quant(A)`` with
       channel-wise pre-scaling followed by 32-element group scaling.

    Before quantisation, each column of the A tile is multiplied by the
    corresponding element of ``channel_scale`` (shape ``(K,)``).  The
    projection path uses the **original** (unscaled) A tile.

    The fusion works because rank == QUANT_GROUP == 32.  Each K-loop step
    loads a ``(BLOCK_M, 32)`` tile of A.  That same tile is used for:
      - one rank-32 outer-product update  ``acc_p += a_tile @ r_tile^T``
      - one complete 32-element quantisation group

    All dimensions and strides are constexpr for full compile-time
    optimization (loop unrolling, address arithmetic, dead code elimination).
    Recompiles per unique (M, K) shape.

    Grid: ``(ceil(M / BLOCK_M),)``  -- 1-D, one program per row-block.
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # (BLOCK_M,)
    offs_r = tl.arange(0, RANK)                          # (RANK,)   == (32,)
    m_mask = offs_m < M                                  # (BLOCK_M,)

    # Accumulator for the projection  P = A @ R^T   shape (BLOCK_M, RANK)
    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    # E2M1 constants
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    num_groups = tl.cdiv(K, QUANT_GROUP)

    for g in range(0, num_groups):
        # --- load A tile  (BLOCK_M, 32) ---
        offs_k = g * QUANT_GROUP + tl.arange(0, QUANT_GROUP)
        a_mask = m_mask[:, None] & (offs_k[None, :] < K)
        a_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0,
        ).to(tl.float16)

        # --- load R tile  (RANK, 32) ---
        r_mask = offs_k[None, :] < K
        r_tile = tl.load(
            R_ptr + offs_r[:, None] * stride_rr + offs_k[None, :] * stride_rk,
            mask=r_mask, other=0.0,
        ).to(tl.float16)
        # r_tile is (RANK, 32),  we need (32, RANK) for the dot  a_tile @ r_tile^T
        # tl.dot expects (BLOCK_M, 32) @ (32, RANK) -> (BLOCK_M, RANK)
        r_tile_t = tl.trans(r_tile)                      # (32, RANK)
        acc_p = tl.dot(a_tile, r_tile_t, acc=acc_p, out_dtype=tl.float32)

        # ===== MXFP4 quantisation of this 32-element group ==================

        # --- channel-wise pre-scaling (quant path only) ---
        cs = tl.load(
            channel_scale_ptr + offs_k * stride_cs,
            mask=offs_k < K, other=1.0,
        ).to(tl.float16)                            # (QUANT_GROUP,)
        a_tile_scaled = a_tile * cs[None, :]         # (BLOCK_M, QUANT_GROUP)

        # --- e8m0 scale ---
        amax = tl.max(tl.abs(a_tile_scaled), axis=1, keep_dims=True)
        amax_i = amax.to(tl.int32, bitcast=True)
        amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax_i.to(tl.float32, bitcast=True)
        scale_exp = tl.log2(amax).floor() - 2
        scale_exp = tl.clamp(scale_exp, min=-127, max=127)
        quant_scale = tl.exp2(-scale_exp)
        scale_e8m0 = scale_exp.to(tl.uint8) + 127

        # --- quantise to e2m1 ---
        qx = a_tile_scaled * quant_scale
        qx_u = qx.to(tl.uint32, bitcast=True)
        s = qx_u & 0x80000000
        e = (qx_u >> 23) & 0xFF
        m_bits = qx_u & 0x7FFFFF

        adj = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
        m_bits = tl.where(e < E8_BIAS, (0x400000 | (m_bits >> 1)) >> adj, m_bits)
        e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        e2m1_tmp = tl.minimum((((e << 2) | (m_bits >> 21)) + 1) >> 1, 0x7)
        e2m1 = ((s >> 28) | e2m1_tmp).to(tl.uint8)

        # --- pack two nibbles per byte ---
        e2m1 = tl.reshape(e2m1, [BLOCK_M, QUANT_GROUP // 2, 2])
        evens, odds = tl.split(e2m1)
        packed = evens | (odds << 4)

        # --- store packed FP4 ---
        out_n = g * QUANT_GROUP // 2 + tl.arange(0, QUANT_GROUP // 2)
        out_mask = m_mask[:, None] & (out_n[None, :] < (K // 2))
        tl.store(
            A_fp4_ptr + offs_m[:, None] * stride_fp4_m + out_n[None, :] * stride_fp4_n,
            packed, mask=out_mask,
        )

        # --- store e8m0 scale ---
        tl.store(
            A_scale_ptr + offs_m[:, None] * stride_sm + g * stride_sn,
            scale_e8m0, mask=m_mask[:, None],
        )

    # ===== store projection result  P (BLOCK_M, RANK)  ======================
    p = acc_p.to(P_ptr.type.element_ty)
    p_mask = m_mask[:, None] & (offs_r[None, :] < RANK)
    tl.store(
        P_ptr + offs_m[:, None] * stride_pm + offs_r[None, :] * stride_pr,
        p, mask=p_mask,
    )


# ===========================================================================
# Kernel 6 -- LoRA+Q: dual GEMM  P @ L^T  +  Q(A) @ Q(W)^T  (fused)
# ===========================================================================

@triton.jit
def loraq_dual_gemm_kernel(
    # Low-rank inputs
    P_ptr,          # (M, RANK) fp16   -- projection from kernel 4
    L_ptr,          # (N, RANK) fp16   -- low-rank factor  (row-major by N)
    # FP4 inputs
    A_fp4_ptr,      # (M, K // 2) uint8 packed e2m1
    A_scale_ptr,    # (M, K // 32) uint8 e8m0
    W_fp4_ptr,      # (K // 2, N) uint8 packed e2m1  **transposed weight**
    W_scale_ptr,    # (N, K // 32) uint8 e8m0         **scale by output channel**
    # Output
    C_ptr,          # (M, N) output
    # Dimensions (constexpr for full compile-time optimization)
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for P (M, RANK)
    stride_pm: tl.constexpr, stride_pr: tl.constexpr,
    # Strides for L (N, RANK)
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    # Strides for A_fp4 (M, K // 2)
    stride_afm: tl.constexpr, stride_afk: tl.constexpr,
    # Strides for A_scale (M, K // 32)
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    # Strides for W_fp4 (K // 2, N)
    stride_wfk: tl.constexpr, stride_wfn: tl.constexpr,
    # Strides for W_scale (N, K // 32)
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    # Strides for C (M, N)
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Compile-time constants
    RANK: tl.constexpr,         # 32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,      # >= 64, multiple of 64 for dot_scaled
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused kernel that computes:

        C[m, n] = (P @ L^T)[m, n]  +  (Q(A) @ Q(W)^T)[m, n]

    in a single output tile, avoiding an extra global-memory round-trip for
    the intermediate results.

    **Low-rank GEMM** (P @ L^T):  K-dim = RANK = 32.  Since RANK fits in
    one tile, this is a single ``tl.dot`` (no K-loop).

    **FP4 GEMM** (Q(A) @ Q(W)^T):  Uses ``tl.dot_scaled`` with e2m1
    format, looping over K in steps of BLOCK_K (>=64).

    Both accumulators (fp32) are summed before the final store.

    All dimensions and strides are constexpr for full compile-time
    optimization (loop unrolling, address arithmetic, dead code elimination).
    Recompiles per unique (M, N, K) shape.

    Grid: ``(ceil(M / BLOCK_M) * ceil(N / BLOCK_N),)`` with L2 swizzle.
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ===== 1. Low-rank GEMM:  P @ L^T  (single dot, K = RANK = 32) ==========
    offs_r = tl.arange(0, RANK)

    # Load P tile  (BLOCK_M, RANK)
    p_mask = (offs_m[:, None] < M) & (offs_r[None, :] < RANK)
    p_tile = tl.load(
        P_ptr + offs_m[:, None] * stride_pm + offs_r[None, :] * stride_pr,
        mask=p_mask, other=0.0,
    )

    # Load L tile  (BLOCK_N, RANK) -- we need L^T -> (RANK, BLOCK_N)
    l_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    l_tile = tl.load(
        L_ptr + offs_n[:, None] * stride_ln + offs_r[None, :] * stride_lr,
        mask=l_mask, other=0.0,
    )
    l_tile_t = tl.trans(l_tile)  # (RANK, BLOCK_N)

    # Single dot:  (BLOCK_M, RANK) @ (RANK, BLOCK_N) -> (BLOCK_M, BLOCK_N)
    acc_lr = tl.dot(p_tile, l_tile_t).to(tl.float32)

    # ===== 2. FP4 GEMM:  Q(A) @ Q(W)^T  via dot_scaled ======================
    # Use modulo-wrapped indices for safe loading (boundary masking at store)
    rm = offs_m % M
    rn = offs_n % N

    rk = tl.arange(0, BLOCK_K // 2)
    rks = tl.arange(0, BLOCK_K // SCALE_GROUP)

    A_BASE = A_fp4_ptr + rm[:, None] * stride_afm + rk[None, :] * stride_afk
    W_BASE = W_fp4_ptr + rk[:, None] * stride_wfk + rn[None, :] * stride_wfn

    A_scale_BASE = A_scale_ptr + rm[:, None] * stride_asm + rks[None, :] * stride_ask
    W_scale_BASE = W_scale_ptr + rn[:, None] * stride_wsn + rks[None, :] * stride_wsk

    acc_q = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        a_scales = tl.load(A_scale_BASE)
        w_scales = tl.load(W_scale_BASE)

        k_remaining = K // 2 - k * (BLOCK_K // 2)
        a_data = tl.load(A_BASE, mask=rk[None, :] < k_remaining, other=0)
        w_data = tl.load(W_BASE, mask=rk[:, None] < k_remaining, other=0)

        acc_q = tl.dot_scaled(a_data, a_scales, "e2m1", w_data, w_scales, "e2m1", acc=acc_q, out_dtype=tl.float32)

        A_BASE += (BLOCK_K // 2) * stride_afk
        W_BASE += (BLOCK_K // 2) * stride_wfk
        A_scale_BASE += (BLOCK_K // SCALE_GROUP) * stride_ask
        W_scale_BASE += (BLOCK_K // SCALE_GROUP) * stride_wsk

    # ===== 3. Sum and store ===================================================
    c = (acc_lr + acc_q).to(C_ptr.type.element_ty)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, c, mask=c_mask)


# ===========================================================================
# Kernel 7 -- LoRaQ: fused FP8/FP4 LoRA+Q with MXFP8 output
#   C_fp8 = q8( fp16(q8(A) @ q8(R)^T) @ fp16(q8(L))^T  +  q8(A) @ q4(W)^T )
# ===========================================================================

@triton.jit
def loraq_fused_q8_kernel(
    # ---- FP8 activation (pre-quantized) ----
    A_fp8_ptr,          # (M, K)     float8_e4m3fn
    A_scale_ptr,        # (M, K//32) uint8 e8m0
    # ---- FP8 low-rank factor R ----
    R_fp8_ptr,          # (RANK, K)     float8_e4m3fn  (row-major)
    R_scale_ptr,        # (RANK, K//32) uint8 e8m0
    # ---- FP8 low-rank factor L ----
    L_fp8_ptr,          # (N, RANK)      float8_e4m3fn
    L_scale_ptr,        # (N, RANK//32)  uint8 e8m0
    # ---- FP4 weight (already transposed: K//2 × N) ----
    W_fp4_ptr,          # (K//2, N) uint8 packed e2m1
    W_scale_ptr,        # (N, K//32) uint8 e8m0
    # ---- optional bias ----
    bias_ptr,           # (N,) float32  (ignored when HAS_BIAS=False)
    # ---- channel-wise quantization scale ----
    channel_scale_ptr,  # (N,) fp16/bf16 — per-column scale before output quant
    # ---- outputs ----
    C_fp8_ptr,          # (M, N)     uint8  (will be viewed as float8_e4m3fn)
    C_scale_ptr,        # (M, N//32) uint8 e8m0
    # ---- dimensions (constexpr for full compile-time optimization) ----
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # ---- strides: A_fp8 (M, K) ----
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # ---- strides: A_scale (M, K//32) ----
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    # ---- strides: R_fp8 (RANK, K) ----
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    # ---- strides: R_scale (RANK, K//32) ----
    stride_rsr: tl.constexpr, stride_rsk: tl.constexpr,
    # ---- strides: L_fp8 (N, RANK) ----
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    # ---- strides: L_scale (N, RANK//32) ----
    stride_lsn: tl.constexpr, stride_lsk: tl.constexpr,
    # ---- strides: W_fp4 (K//2, N) ----
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    # ---- strides: W_scale (N, K//32) ----
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    # ---- strides: C_fp8 (M, N) ----
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # ---- strides: C_scale (M, N//32) ----
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    # ---- stride: channel_scale (N,) ----
    stride_cs: tl.constexpr,
    # ---- compile-time constants ----
    HAS_BIAS: tl.constexpr,
    RANK: tl.constexpr,            # 64
    BLOCK_M: tl.constexpr,         # 128
    BLOCK_N: tl.constexpr,         # 128
    BLOCK_K: tl.constexpr,         # 64  (minimum for dot_scaled with e2m1)
    GROUP_SIZE_M: tl.constexpr,    # 8
):
    """
    Fused kernel computing:

        C_fp8 = MXFP8_quant(
            channel_scale * (
                fp16(q8(A) @ q8(R)^T) @ dequant_fp16(q8(L))^T
                + q8(A) @ q4(W)^T
                [+ bias]
            )
        )

    Three phases per output tile (BLOCK_M × BLOCK_N):

    Phase 1  Fused K-loop — A×R^T via dot_scaled("e4m3","e4m3") and
             A×W^T via dot_scaled("e4m3","e2m1").  A tiles loaded once,
             reused for both sub-GEMMs.

    Phase 2  P × L^T — P (fp32→fp16) times L (fp8→fp16) via tl.dot.
             Mixed-precision by design: keeps P in fp16 (higher fidelity
             than re-quantizing to fp8) and L dequantized from fp8 in-
             register (half the bandwidth of fp16 storage).

    Phase 3  Sum + optional bias + channel-wise pre-scaling +
             in-register MXFP8 quantization.
             Groups of 32 along N each get an e8m0 scale.

    All dimensions and strides are constexpr for full compile-time
    optimization (loop unrolling, address arithmetic, dead code elimination).
    Recompiles per unique (M, N, K) shape.

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly grouped swizzle (identical to other kernels)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)

    # Modulo-wrapped row/col indices for safe boundary-free loading
    rm = offs_m % M
    rn = offs_n % N

    # ===== Phase 1 — Fused K-loop  (A×R^T  and  A×W^T) =====================

    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)     # A @ R^T
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # A @ W^T

    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, BLOCK_K // SCALE_GROUP)

        # ---- Load A fp8 tile (BLOCK_M, BLOCK_K) ----
        a_tile = tl.load(
            A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=offs_k[None, :] < K,
            other=0.0,
        )
        # ---- Load A scale (BLOCK_M, BLOCK_K//32) ----
        a_scale = tl.load(
            A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
            mask=offs_kg[None, :] < (K // SCALE_GROUP),
            other=0,
        )

        # ---- A × R^T:  dot_scaled("e4m3", "e4m3") ----
        # R is (RANK, K) row-major → load transposed as (BLOCK_K, RANK)
        r_tile = tl.load(
            R_fp8_ptr + offs_r[None, :] * stride_rr + offs_k[:, None] * stride_rk,
            mask=offs_k[:, None] < K,
            other=0.0,
        )
        # R scale (RANK, K//32) — needed as (RANK, BLOCK_K//32) for dot_scaled
        r_scale = tl.load(
            R_scale_ptr + offs_r[:, None] * stride_rsr + offs_kg[None, :] * stride_rsk,
            mask=offs_kg[None, :] < (K // SCALE_GROUP),
            other=0,
        )

        acc_p = tl.dot_scaled(a_tile, a_scale, "e4m3",
                               r_tile, r_scale, "e4m3",
                               acc=acc_p, out_dtype=tl.float32)

        # ---- A × W^T:  dot_scaled("e4m3", "e2m1") ----
        # W is (K//2, N) packed fp4 → load (BLOCK_K//2, BLOCK_N)
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_tile = tl.load(
            W_fp4_ptr + offs_k_packed[:, None] * stride_wk + rn[None, :] * stride_wn,
            mask=offs_k_packed[:, None] < (K // 2),
            other=0,
        )
        # W scale (N, K//32) → load (BLOCK_N, BLOCK_K//32)
        w_scale = tl.load(
            W_scale_ptr + rn[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk,
            mask=offs_kg[None, :] < (K // SCALE_GROUP),
            other=0,
        )

        result = tl.dot_scaled(a_tile, a_scale, "e4m3",
                               w_tile, w_scale, "e2m1",
                               acc=result, out_dtype=tl.float32)

    # ===== Phase 2 — P × L^T  (fp16 × dequant-fp8→fp16) ====================

    # Load L fp8 tile (BLOCK_N, RANK)
    l_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    l_fp8 = tl.load(
        L_fp8_ptr + offs_n[:, None] * stride_ln + offs_r[None, :] * stride_lr,
        mask=l_mask, other=0.0,
    )

    # Load L scales via gather expansion:
    #   L_scale is (N, RANK//32).  offs_r//32 maps each rank element to
    #   its scale-group index (0 for ranks 0-31, 1 for ranks 32-63).
    l_scale_group = offs_r // SCALE_GROUP   # (RANK,) compile-time pattern
    l_scale = tl.load(
        L_scale_ptr + offs_n[:, None] * stride_lsn
                    + l_scale_group[None, :] * stride_lsk,
        mask=offs_n[:, None] < N,
        other=127,  # scale 127 → multiplier 1.0
    )   # (BLOCK_N, RANK) — each element carries its group's scale

    # Dequant L to fp16:  fp8_val × 2^(scale - 127)
    l_fp16 = (l_fp8
              * tl.exp2((l_scale - 127.0))).to(tl.float16)   # (BLOCK_N, RANK)

    # Single dot: (BLOCK_M, RANK) @ (RANK, BLOCK_N) → (BLOCK_M, BLOCK_N)
    result = tl.dot(acc_p.to(tl.float16), tl.trans(l_fp16), acc=result,out_dtype=tl.float32)

    # ===== Phase 3 — Sum + bias + channel-wise scaling + MXFP8 quant ========


    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        result = result + bias[None, :]

    # ---- channel-wise pre-scaling before quantization ----
    cs = tl.load(
        channel_scale_ptr + offs_n * stride_cs,
        mask=offs_n < N, other=1.0,
    )                                    # (BLOCK_N,)
    result = result.to(tl.float16) * cs[None, :]                       # (BLOCK_M, BLOCK_N)

    # ---- group-wise MXFP8 quantization ----
    N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP

    # Reshape to (BLOCK_M, N_GROUPS, 32) for per-group amax
    result_3d = tl.reshape(result, [BLOCK_M, N_GROUPS, SCALE_GROUP])
    amax = tl.max(tl.abs(result_3d), axis=2)   # (BLOCK_M, N_GROUPS)

    # Round amax up to nearest power of two
    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)

    # e8m0 scale:  floor(log2(amax)) - 7   (max e4m3 ≈ 448 ≈ 2^8.8)
    scale_exp = tl.log2(amax).floor() - 7
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    scale_e8m0 = scale_exp.to(tl.uint8) + 127          # (BLOCK_M, N_GROUPS)

    # Expand normalisation factor to (BLOCK_M, N_GROUPS, 1) for broadcast
    quant_scale = tl.exp2(-scale_exp)                   # (BLOCK_M, N_GROUPS)
    quant_scale_3d = tl.reshape(quant_scale, [BLOCK_M, N_GROUPS, 1])

    # Normalise each 32-element group
    qx_3d = result_3d * quant_scale_3d                  # broadcast last dim
    qx = tl.reshape(qx_3d, [BLOCK_M, BLOCK_N])
    qx = tl.clamp(qx, min=-448.0, max=448.0)

    fp8_out = qx.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    # ---- store fp8 data (BLOCK_M, BLOCK_N) ----
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_fp8_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        fp8_out, mask=c_mask,
    )

    # ---- store e8m0 scales (BLOCK_M, N_GROUPS) ----
    offs_ng = pid_n * N_GROUPS + tl.arange(0, N_GROUPS)
    s_mask = (offs_m[:, None] < M) & (offs_ng[None, :] < (N // SCALE_GROUP))
    tl.store(
        C_scale_ptr + offs_m[:, None] * stride_csm + offs_ng[None, :] * stride_csn,
        scale_e8m0, mask=s_mask,
    )


# ===========================================================================
# Breakdown kernels for K8 path analysis
# ===========================================================================

# -- K_lowrank: fused low-rank branch (A×R^T → P quant → P×L^T → fp16 out) --

@triton.jit
def loraq_lowrank_branch_kernel(
    A_fp8_ptr, A_scale_ptr,
    R_fp8_ptr, R_scale_ptr,
    L_fp8_ptr, L_scale_ptr,
    C_ptr,  # (M, N) fp16 output
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    stride_rsr: tl.constexpr, stride_rsk: tl.constexpr,
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    stride_lsn: tl.constexpr, stride_lsk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Low-rank branch only: A_fp8×R_fp8^T → P → quant(P) → P_fp8×L_fp8^T → fp16."""
    SCALE_GROUP: tl.constexpr = 32
    RANK_GROUPS: tl.constexpr = RANK // SCALE_GROUP

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)
    rm = offs_m % M

    # Phase 1: A × R^T via dot_scaled
    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, BLOCK_K // SCALE_GROUP)

        a_tile = tl.load(A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
                         mask=offs_k[None, :] < K, other=0.0)
        a_scale = tl.load(A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)
        r_tile = tl.load(R_fp8_ptr + offs_r[None, :] * stride_rr + offs_k[:, None] * stride_rk,
                         mask=offs_k[:, None] < K, other=0.0)
        r_scale = tl.load(R_scale_ptr + offs_r[:, None] * stride_rsr + offs_kg[None, :] * stride_rsk,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)
        acc_p = tl.dot_scaled(a_tile, a_scale, "e4m3", r_tile, r_scale, "e4m3",
                               acc=acc_p, out_dtype=tl.float32)

    # Phase 2: In-register MXFP8 quant of P + P×L^T via dot_scaled
    p_3d = tl.reshape(acc_p, [BLOCK_M, RANK_GROUPS, SCALE_GROUP])
    p_amax = tl.max(tl.abs(p_3d), axis=2)
    p_amax_i = p_amax.to(tl.int32, bitcast=True)
    p_amax_i = (p_amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    p_amax = p_amax_i.to(tl.float32, bitcast=True)
    p_scale_exp = tl.log2(p_amax).floor() - 7
    p_scale_exp = tl.clamp(p_scale_exp, min=-127, max=127)
    p_scale_e8m0 = p_scale_exp.to(tl.uint8) + 127
    p_quant = tl.exp2(-p_scale_exp)
    p_quant_3d = tl.reshape(p_quant, [BLOCK_M, RANK_GROUPS, 1])
    qp_3d = p_3d * p_quant_3d
    qp = tl.reshape(qp_3d, [BLOCK_M, RANK])
    qp = tl.clamp(qp, min=-448.0, max=448.0)
    p_fp8 = qp.to(tl.float8e4nv)

    # Load L as (RANK, BLOCK_N) + scale
    l_tile = tl.load(L_fp8_ptr + offs_n[None, :] * stride_ln + offs_r[:, None] * stride_lr,
                     mask=(offs_n[None, :] < N) & (offs_r[:, None] < RANK), other=0.0)
    offs_rk = tl.arange(0, RANK_GROUPS)
    l_scale = tl.load(L_scale_ptr + offs_n[:, None] * stride_lsn + offs_rk[None, :] * stride_lsk,
                      mask=offs_n[:, None] < N, other=0)

    result = tl.dot_scaled(p_fp8, p_scale_e8m0, "e4m3", l_tile, l_scale, "e4m3")

    # Store fp16
    c = result.to(tl.float16)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=c_mask)


# -- K_residual: standalone fp8×fp4 GEMM (A×W^T → fp16 out) --

@triton.jit
def loraq_residual_branch_kernel(
    A_fp8_ptr, A_scale_ptr,
    W_fp4_ptr, W_scale_ptr,
    C_ptr,  # (M, N) fp16 output
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Standalone A_fp8 × W_fp4^T → fp16 output via dot_scaled(e4m3, e2m1)."""
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = offs_m % M
    rn = offs_n % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, BLOCK_K // SCALE_GROUP)

        a_tile = tl.load(A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
                         mask=offs_k[None, :] < K, other=0.0)
        a_scale = tl.load(A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)

        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_tile = tl.load(W_fp4_ptr + offs_k_packed[:, None] * stride_wk + rn[None, :] * stride_wn,
                         mask=offs_k_packed[:, None] < (K // 2), other=0)
        w_scale = tl.load(W_scale_ptr + rn[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)

        acc = tl.dot_scaled(a_tile, a_scale, "e4m3", w_tile, w_scale, "e2m1",
                            acc=acc, out_dtype=tl.float32)

    c = acc.to(tl.float16)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=c_mask)


# -- K_sum_quant: fp16 add + MXFP8 output quant --

@triton.jit
def loraq_sum_quant_kernel(
    A_ptr, B_ptr,  # (M, N) fp16 inputs
    C_fp8_ptr, C_scale_ptr,  # MXFP8 output
    M: tl.constexpr, N: tl.constexpr,
    stride_am: tl.constexpr, stride_an: tl.constexpr,
    stride_bm: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Add two fp16 tensors + MXFP8 quantize the result."""
    SCALE_GROUP: tl.constexpr = 32

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask, other=0.0)
    result = (a + b).to(tl.float16)

    # MXFP8 quant
    N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP
    result_3d = tl.reshape(result, [BLOCK_M, N_GROUPS, SCALE_GROUP])
    amax = tl.max(tl.abs(result_3d), axis=2)
    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)
    scale_exp = tl.log2(amax).floor() - 7
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    scale_e8m0 = scale_exp.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_exp)
    quant_scale_3d = tl.reshape(quant_scale, [BLOCK_M, N_GROUPS, 1])
    qx_3d = result_3d * quant_scale_3d
    qx = tl.reshape(qx_3d, [BLOCK_M, BLOCK_N])
    qx = tl.clamp(qx, min=-448.0, max=448.0)
    fp8_out = qx.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    tl.store(C_fp8_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, fp8_out, mask=mask)

    offs_ng = pid_n * N_GROUPS + tl.arange(0, N_GROUPS)
    s_mask = (offs_m[:, None] < M) & (offs_ng[None, :] < (N // SCALE_GROUP))
    tl.store(C_scale_ptr + offs_m[:, None] * stride_csm + offs_ng[None, :] * stride_csn,
             scale_e8m0, mask=s_mask)


# ===========================================================================
# Kernel 9 -- LoRaQ.7: like K7 but Phase 3 uses v_cvt_scalef32_pk_fp8_f16
#   Hardware FP8 quantization via inline assembly (MI350 gfx950)
# ===========================================================================

@triton.jit
def loraq_fused_q8_hwquant_kernel(
    # ---- IDENTICAL interface to K7 ----
    A_fp8_ptr, A_scale_ptr,
    R_fp8_ptr, R_scale_ptr,
    L_fp8_ptr, L_scale_ptr,
    W_fp4_ptr, W_scale_ptr,
    bias_ptr, channel_scale_ptr,
    C_fp8_ptr, C_scale_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    stride_rsr: tl.constexpr, stride_rsk: tl.constexpr,
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    stride_lsn: tl.constexpr, stride_lsk: tl.constexpr,
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    stride_cs: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    LoRaQ.7 — identical to K7 but Phase 3 uses MI350 hardware instruction
    ``v_cvt_scalef32_pk_fp8_f16`` for MXFP8 output quantization.

    The hardware instruction takes packed fp16 pairs + e8m0 scale exponent
    and produces packed fp8 output in a single ALU op, eliminating the
    manual normalize→clamp→convert pipeline.

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)
    rm = offs_m % M
    rn = offs_n % N

    # ===== Phase 1 — Fused K-loop (identical to K7) =========================
    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, BLOCK_K // SCALE_GROUP)

        a_tile = tl.load(A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
                         mask=offs_k[None, :] < K, other=0.0)
        a_scale = tl.load(A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)

        r_tile = tl.load(R_fp8_ptr + offs_r[None, :] * stride_rr + offs_k[:, None] * stride_rk,
                         mask=offs_k[:, None] < K, other=0.0)
        r_scale = tl.load(R_scale_ptr + offs_r[:, None] * stride_rsr + offs_kg[None, :] * stride_rsk,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)

        acc_p = tl.dot_scaled(a_tile, a_scale, "e4m3", r_tile, r_scale, "e4m3",
                               acc=acc_p, out_dtype=tl.float32)

        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_tile = tl.load(W_fp4_ptr + offs_k_packed[:, None] * stride_wk + rn[None, :] * stride_wn,
                         mask=offs_k_packed[:, None] < (K // 2), other=0)
        w_scale = tl.load(W_scale_ptr + rn[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk,
                          mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0)

        result = tl.dot_scaled(a_tile, a_scale, "e4m3", w_tile, w_scale, "e2m1",
                                acc=result, out_dtype=tl.float32)

    # ===== Phase 2 — P × L^T (identical to K7) ==============================
    l_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    l_fp8 = tl.load(L_fp8_ptr + offs_n[:, None] * stride_ln + offs_r[None, :] * stride_lr,
                    mask=l_mask, other=0.0)
    l_scale_group = offs_r // SCALE_GROUP
    l_scale = tl.load(L_scale_ptr + offs_n[:, None] * stride_lsn + l_scale_group[None, :] * stride_lsk,
                      mask=offs_n[:, None] < N, other=127)
    l_fp16 = (l_fp8 * tl.exp2((l_scale - 127.0))).to(tl.float16)
    result = tl.dot(acc_p.to(tl.float16), tl.trans(l_fp16), acc=result, out_dtype=tl.float32)

    # ===== Phase 3 — HW MXFP8 quant via v_cvt_scalef32_pk_fp8_f16 ==========

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        result = result + bias[None, :]

    cs = tl.load(channel_scale_ptr + offs_n * stride_cs, mask=offs_n < N, other=1.0)
    result_fp16 = (result.to(tl.float16) * cs[None, :]).to(tl.float16)

    N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP
    PAIRS_PER_GROUP: tl.constexpr = SCALE_GROUP // 2  # 16 pairs per group of 32

    # ---- Compute per-group e8m0 scale (same amax logic as K7) ----
    result_3d = tl.reshape(result_fp16, [BLOCK_M, N_GROUPS, SCALE_GROUP])
    amax = tl.max(tl.abs(result_3d), axis=2).to(tl.float16)

    # Extract fp16 exponent for e8m0 scale (offset=8 for e4m3 max≈448≈2^8.8)
    amax_bits = amax.to(tl.uint16, bitcast=True)
    bf16_exp = (amax_bits >> 10) & 0x1F  # fp16 has 5-bit exponent
    # Convert fp16 biased exp (bias=15) to e8m0 biased exp (bias=127):
    # e8m0 = fp16_exp - 15 + 127 - 8 = fp16_exp + 104
    # But we need the full 8-bit range. Use fp32 path instead:
    amax_f32 = amax.to(tl.float32)
    amax_u32 = amax_f32.to(tl.uint32, bitcast=True)
    # Round amax up to nearest power of 2
    amax_u32 = (amax_u32 + 0x200000).to(tl.uint32) & 0xFF800000
    # Extract fp32 biased exponent (8-bit, bias=127)
    fp32_exp = (amax_u32 >> 23) & 0xFF
    # e8m0 scale = fp32_exp - 8 (offset for e4m3 max range)
    scale_e8m0 = tl.maximum(fp32_exp.to(tl.int32) - 8, 0)
    scale_e8m0 = tl.minimum(scale_e8m0, 255).to(tl.uint8)  # (BLOCK_M, N_GROUPS)

    # ---- Store e8m0 scales ----
    offs_ng = pid_n * N_GROUPS + tl.arange(0, N_GROUPS)
    s_mask = (offs_m[:, None] < M) & (offs_ng[None, :] < (N // SCALE_GROUP))
    tl.store(C_scale_ptr + offs_m[:, None] * stride_csm + offs_ng[None, :] * stride_csn,
             scale_e8m0, mask=s_mask)

    # ---- Pack fp16 pairs and call HW conversion ----
    # Reshape result to pairs: (BLOCK_M, BLOCK_N//2, 2)
    result_pairs = tl.reshape(result_fp16, (BLOCK_M, BLOCK_N // 2, 2))
    x_lo, x_hi = tl.split(result_pairs)
    x_lo = tl.reshape(x_lo, (BLOCK_M, BLOCK_N // 2))
    x_hi = tl.reshape(x_hi, (BLOCK_M, BLOCK_N // 2))

    x_lo_u16 = x_lo.to(tl.uint16, bitcast=True)
    x_hi_u16 = x_hi.to(tl.uint16, bitcast=True)
    x_pk = (x_lo_u16.to(tl.uint32) | (x_hi_u16.to(tl.uint32) << 16)).to(tl.float32, bitcast=True)

    # Broadcast scale to pairs: each group of 16 pairs shares one scale
    scale_broad = tl.broadcast_to(
        tl.reshape(scale_e8m0.to(tl.uint32), (BLOCK_M, N_GROUPS, 1)),
        (BLOCK_M, N_GROUPS, PAIRS_PER_GROUP),
    )
    scale_flat = tl.reshape(scale_broad, (BLOCK_M, BLOCK_N // 2))
    scale_f32 = scale_flat << 23

    # HW FP8 conversion (RTNE)
    fp8_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_f16 $0, $1, $2",
        "=v,v,v",
        args=[x_pk, scale_f32],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )

    # ---- Store fp8 output (split uint16 into 2 uint8 bytes) ----
    fp8_lo = fp8_packed.to(tl.uint8)
    fp8_hi = (fp8_packed >> 8).to(tl.uint8)

    even_offs = tl.arange(0, BLOCK_N // 2) * 2
    odd_offs = even_offs + 1
    base_n = pid_n * BLOCK_N

    c_mask_even = (offs_m[:, None] < M) & ((base_n + even_offs)[None, :] < N)
    c_mask_odd = (offs_m[:, None] < M) & ((base_n + odd_offs)[None, :] < N)

    tl.store(C_fp8_ptr + offs_m[:, None] * stride_cm + (base_n + even_offs)[None, :] * stride_cn,
             fp8_lo, mask=c_mask_even)
    tl.store(C_fp8_ptr + offs_m[:, None] * stride_cm + (base_n + odd_offs)[None, :] * stride_cn,
             fp8_hi, mask=c_mask_odd)


# ===========================================================================
# Kernel 13 -- LoRaQ.6: like K7 but L and R are fp16 (no fp8 quantization)
#   C_fp8 = q8( fp16(dequant(A_fp8)) @ R_fp16^T @ L_fp16^T
#             + q8(A) @ q4(W)^T )
# ===========================================================================

@triton.jit
def loraq_fused_q8_fp16lr_kernel(
    # ---- FP8 activation (pre-quantized) ----
    A_fp8_ptr,          # (M, K)     float8_e4m3fn
    A_scale_ptr,        # (M, K//32) uint8 e8m0
    # ---- FP16 low-rank factor R ----
    R_ptr,              # (RANK, K)  fp16  (row-major)
    # ---- FP16 low-rank factor L ----
    L_ptr,              # (N, RANK)  fp16
    # ---- FP4 weight (already transposed: K//2 × N) ----
    W_fp4_ptr,          # (K//2, N) uint8 packed e2m1
    W_scale_ptr,        # (N, K//32) uint8 e8m0
    # ---- optional bias ----
    bias_ptr,           # (N,) float32  (ignored when HAS_BIAS=False)
    # ---- channel-wise quantization scale ----
    channel_scale_ptr,  # (N,) fp16 — per-column scale before output quant
    # ---- outputs ----
    C_fp8_ptr,          # (M, N)     uint8
    C_scale_ptr,        # (M, N//32) uint8 e8m0
    # ---- dimensions ----
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # ---- strides: A_fp8 (M, K) ----
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # ---- strides: A_scale (M, K//32) ----
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    # ---- strides: R (RANK, K) ----
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    # ---- strides: L (N, RANK) ----
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    # ---- strides: W_fp4 (K//2, N) ----
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    # ---- strides: W_scale (N, K//32) ----
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    # ---- strides: C_fp8 (M, N) ----
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # ---- strides: C_scale (M, N//32) ----
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    # ---- stride: channel_scale (N,) ----
    stride_cs: tl.constexpr,
    # ---- compile-time constants ----
    HAS_BIAS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    LoRaQ.6 — like K7 but L and R are stored as fp16 (not fp8+scale).

    Computes:
        C_fp8 = MXFP8_quant(
            channel_scale * (
                dequant_fp16(A_fp8) @ R_fp16^T @ L_fp16^T
                + A_fp8 @ W_fp4^T
                [+ bias]
            )
        )

    Phase 1  K-loop:
             - A×R^T: dequant A_fp8→fp16 in-register, then tl.dot(A_fp16, R_fp16^T)
             - A×W^T: dot_scaled(A_fp8, W_fp4) — same as K7
             Both reuse the same A_fp8 tile + A_scale per iteration.

    Phase 2  P × L^T: tl.dot(P_fp16, L_fp16^T) — L loaded directly as fp16.

    Phase 3  Same MXFP8 output quant as K7.

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
    """
    SCALE_GROUP: tl.constexpr = 32
    K_GROUPS: tl.constexpr = BLOCK_K // SCALE_GROUP

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)

    rm = offs_m % M
    rn = offs_n % N

    # ===== Phase 1 — Fused K-loop  (A×R^T fp16 and A×W^T dot_scaled) =======

    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, K_GROUPS)

        # ---- Load A fp8 tile (BLOCK_M, BLOCK_K) ----
        a_fp8 = tl.load(
            A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=offs_k[None, :] < K, other=0.0,
        )
        # ---- Load A scale (BLOCK_M, K_GROUPS) ----
        a_scale = tl.load(
            A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
            mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0,
        )

        # ---- Dequant A_fp8 → fp16 for the low-rank branch ----
        # Reshape to (BLOCK_M, K_GROUPS, 32), multiply by per-group scale
        a_fp8_3d = tl.reshape(a_fp8.to(tl.float32), [BLOCK_M, K_GROUPS, SCALE_GROUP])
        a_scale_3d = tl.reshape(
            tl.exp2((a_scale.to(tl.float32) - 127.0)),
            [BLOCK_M, K_GROUPS, 1],
        )
        a_fp16 = tl.reshape(a_fp8_3d * a_scale_3d, [BLOCK_M, BLOCK_K]).to(tl.float16)

        # ---- A × R^T via tl.dot fp16 ----
        # R is (RANK, K) fp16 → load as (BLOCK_K, RANK) transposed
        r_tile = tl.load(
            R_ptr + offs_r[None, :] * stride_rr + offs_k[:, None] * stride_rk,
            mask=offs_k[:, None] < K, other=0.0,
        ).to(tl.float16)   # (BLOCK_K, RANK)

        acc_p = tl.dot(a_fp16, r_tile, acc=acc_p, out_dtype=tl.float32)

        # ---- A × W^T via dot_scaled (same as K7) ----
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_tile = tl.load(
            W_fp4_ptr + offs_k_packed[:, None] * stride_wk + rn[None, :] * stride_wn,
            mask=offs_k_packed[:, None] < (K // 2), other=0,
        )
        w_scale = tl.load(
            W_scale_ptr + rn[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk,
            mask=offs_kg[None, :] < (K // SCALE_GROUP), other=0,
        )
        result = tl.dot_scaled(a_fp8, a_scale, "e4m3",
                                w_tile, w_scale, "e2m1",
                                acc=result, out_dtype=tl.float32)

    # ===== Phase 2 — P × L^T  (fp16 × fp16, no dequant needed) =============

    # Load L tile (BLOCK_N, RANK) fp16
    l_mask = (offs_n[:, None] < N) & (offs_r[None, :] < RANK)
    l_tile = tl.load(
        L_ptr + offs_n[:, None] * stride_ln + offs_r[None, :] * stride_lr,
        mask=l_mask, other=0.0,
    ).to(tl.float16)

    # Single dot: (BLOCK_M, RANK) @ (RANK, BLOCK_N) → (BLOCK_M, BLOCK_N)
    result = tl.dot(acc_p.to(tl.float16), tl.trans(l_tile),
                    acc=result, out_dtype=tl.float32)

    # ===== Phase 3 — bias + channel-wise scaling + MXFP8 quant =============

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        result = result + bias[None, :]

    cs = tl.load(
        channel_scale_ptr + offs_n * stride_cs,
        mask=offs_n < N, other=1.0,
    )
    result = result.to(tl.float16) * cs[None, :]

    N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP

    result_3d = tl.reshape(result, [BLOCK_M, N_GROUPS, SCALE_GROUP])
    amax = tl.max(tl.abs(result_3d), axis=2)

    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)

    scale_exp = tl.log2(amax).floor() - 7
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    scale_e8m0 = scale_exp.to(tl.uint8) + 127

    quant_scale = tl.exp2(-scale_exp)
    quant_scale_3d = tl.reshape(quant_scale, [BLOCK_M, N_GROUPS, 1])

    qx_3d = result_3d * quant_scale_3d
    qx = tl.reshape(qx_3d, [BLOCK_M, BLOCK_N])
    qx = tl.clamp(qx, min=-448.0, max=448.0)

    fp8_out = qx.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_fp8_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        fp8_out, mask=c_mask,
    )

    offs_ng = pid_n * N_GROUPS + tl.arange(0, N_GROUPS)
    s_mask = (offs_m[:, None] < M) & (offs_ng[None, :] < (N // SCALE_GROUP))
    tl.store(
        C_scale_ptr + offs_m[:, None] * stride_csm + offs_ng[None, :] * stride_csn,
        scale_e8m0, mask=s_mask,
    )


# ===========================================================================
# Kernel 8 -- LoRaQ (FP8 variant): Phase 2 uses dot_scaled("e4m3","e4m3")
#   C_fp8 = q8( q8(q8(A) @ q8(R)^T) @ q8(L)^T  +  q8(A) @ q4(W)^T )
# ===========================================================================

@triton.jit
def loraq_fused_q8_scaled_kernel(
    # ---- FP8 activation (pre-quantized) ----
    A_fp8_ptr,          # (M, K)     float8_e4m3fn
    A_scale_ptr,        # (M, K//32) uint8 e8m0
    # ---- FP8 low-rank factor R ----
    R_fp8_ptr,          # (RANK, K)     float8_e4m3fn  (row-major)
    R_scale_ptr,        # (RANK, K//32) uint8 e8m0
    # ---- FP8 low-rank factor L ----
    L_fp8_ptr,          # (N, RANK)      float8_e4m3fn
    L_scale_ptr,        # (N, RANK//32)  uint8 e8m0
    # ---- FP4 weight (already transposed: K//2 × N) ----
    W_fp4_ptr,          # (K//2, N) uint8 packed e2m1
    W_scale_ptr,        # (N, K//32) uint8 e8m0
    # ---- optional bias ----
    bias_ptr,           # (N,) float32  (ignored when HAS_BIAS=False)
    # ---- channel-wise quantization scale ----
    channel_scale_ptr,  # (N,) fp16/bf16 — per-column scale before output quant
    # ---- outputs ----
    C_fp8_ptr,          # (M, N)     uint8  (will be viewed as float8_e4m3fn)
    C_scale_ptr,        # (M, N//32) uint8 e8m0
    # ---- dimensions (runtime — constexpr causes MLIR crash for this kernel) ----
    M, N, K,
    # ---- strides: A_fp8 (M, K) ----
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # ---- strides: A_scale (M, K//32) ----
    stride_asm: tl.constexpr, stride_ask: tl.constexpr,
    # ---- strides: R_fp8 (RANK, K) ----
    stride_rr: tl.constexpr, stride_rk: tl.constexpr,
    # ---- strides: R_scale (RANK, K//32) ----
    stride_rsr: tl.constexpr, stride_rsk: tl.constexpr,
    # ---- strides: L_fp8 (N, RANK) ----
    stride_ln: tl.constexpr, stride_lr: tl.constexpr,
    # ---- strides: L_scale (N, RANK//32) ----
    stride_lsn: tl.constexpr, stride_lsk: tl.constexpr,
    # ---- strides: W_fp4 (K//2, N) ----
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    # ---- strides: W_scale (N, K//32) ----
    stride_wsn: tl.constexpr, stride_wsk: tl.constexpr,
    # ---- strides: C_fp8 (M, N) ----
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # ---- strides: C_scale (M, N//32) ----
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    # ---- stride: channel_scale (N,) ----
    stride_cs: tl.constexpr,
    # ---- compile-time constants ----
    HAS_BIAS: tl.constexpr,
    RANK: tl.constexpr,            # 64
    BLOCK_M: tl.constexpr,         # 128
    BLOCK_N: tl.constexpr,         # 128
    BLOCK_K: tl.constexpr,         # 64
    GROUP_SIZE_M: tl.constexpr,    # 8
):
    """
    Fused kernel computing (fully-fp8 Phase 2 variant):

        C_fp8 = MXFP8_quant(
            channel_scale * (
                q8(q8(A) @ q8(R)^T) @ q8(L)^T
                + q8(A) @ q4(W)^T
                [+ bias]
            )
        )

    Identical to ``loraq_fused_q8_kernel`` (Kernel 7) except Phase 2:

    Phase 2  P × L^T — P is quantized to MXFP8 in-register (fp32→fp8),
             then dot_scaled("e4m3","e4m3") is used.  This trades some
             precision (extra P quantization) for potentially higher
             throughput via the hardware-accelerated scaled-dot path.
             RANK=64 satisfies the K≥64 requirement for dot_scaled.

    Phases 1 and 3 are identical to Kernel 7.

    NOTE: M, N, K are kept as runtime parameters (not constexpr) because
    the in-register MXFP8 quantization in Phase 2 causes an MLIR compiler
    crash (ConvertTritonAMDGPUToLLVM) when combined with constexpr dims.
    Strides are constexpr for address arithmetic folding.

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly grouped swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)

    rm = offs_m % M
    rn = offs_n % N

    # ===== Phase 1 — Fused K-loop  (A×R^T  and  A×W^T) =====================
    # (identical to Kernel 7)

    acc_p = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    acc_q = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    loop_k = tl.cdiv(K, BLOCK_K)

    for k in range(0, loop_k):
        k0 = k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // SCALE_GROUP) + tl.arange(0, BLOCK_K // SCALE_GROUP)
        
        s_mask = offs_kg[None, :] < (K // SCALE_GROUP)
        
        a_tile = tl.load(
            A_fp8_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=offs_k[None, :] < K, other=0.0,
        )
        a_scale = tl.load(
            A_scale_ptr + rm[:, None] * stride_asm + offs_kg[None, :] * stride_ask,
            mask=s_mask, other=0,
        )

        r_tile = tl.load(
            R_fp8_ptr + offs_r[None, :] * stride_rr + offs_k[:, None] * stride_rk,
            mask=offs_k[:, None] < K, other=0.0,
        )
        r_scale = tl.load(
            R_scale_ptr + offs_r[:, None] * stride_rsr + offs_kg[None, :] * stride_rsk,
            mask=s_mask, other=0,
        )
        acc_p = tl.dot_scaled(a_tile, a_scale, "e4m3",
                               r_tile, r_scale, "e4m3",
                               acc=acc_p, out_dtype=tl.float32)

        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_tile = tl.load(
            W_fp4_ptr + offs_k_packed[:, None] * stride_wk + rn[None, :] * stride_wn,
            mask=offs_k_packed[:, None] < (K // 2), other=0,
        )
        w_scale = tl.load(
            W_scale_ptr + rn[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk,
            mask=s_mask, other=0,
        )
        acc_q = tl.dot_scaled(a_tile, a_scale, "e4m3",
                               w_tile, w_scale, "e2m1",
                               acc=acc_q, out_dtype=tl.float32)

    # ===== Phase 2 — P × L^T  via dot_scaled("e4m3","e4m3") ================
    #
    # Key difference from Kernel 7: P is quantized to MXFP8 in-register,
    # then dot_scaled is used instead of tl.dot with fp16.

    RANK_GROUPS: tl.constexpr = RANK // SCALE_GROUP   # 64 // 32 = 2
    E8_BIAS: tl.constexpr = 127
    E4_BIAS: tl.constexpr = 7

    # ---- In-register MXFP8 quantisation of P (BLOCK_M, RANK) ----
    p_3d = tl.reshape(acc_p, [BLOCK_M, RANK_GROUPS, SCALE_GROUP])
    p_amax = tl.max(tl.abs(p_3d), axis=2)              # (BLOCK_M, RANK_GROUPS)

    p_amax_i = p_amax.to(tl.int32, bitcast=True)
    p_amax_i = (p_amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    p_amax = p_amax_i.to(tl.float32, bitcast=True)

    p_scale_exp = tl.log2(p_amax).floor() - 7
    p_scale_exp = tl.clamp(p_scale_exp, min=-127, max=127)
    p_scale_e8m0 = p_scale_exp.to(tl.uint8) + 127      # (BLOCK_M, RANK_GROUPS)

    p_quant = tl.exp2(-p_scale_exp)                     # (BLOCK_M, RANK_GROUPS)
    p_quant_3d = tl.reshape(p_quant, [BLOCK_M, RANK_GROUPS, 1])

    qp_3d = p_3d * p_quant_3d
    qp = tl.reshape(qp_3d, [BLOCK_M, RANK])
    qp = tl.clamp(qp, min=-448.0, max=448.0)
    p_fp8 = qp.to(tl.float8e4nv)  # (BLOCK_M, RANK) in fp8 format, still as float32
    
    # ---- Load L as (RANK, BLOCK_N) via transposed indexing ----
    l_tile = tl.load(
        L_fp8_ptr + offs_n[None, :] * stride_ln + offs_r[:, None] * stride_lr,
        mask=(offs_n[None, :] < N) & (offs_r[:, None] < RANK),
        other=0.0,
    )   # (RANK, BLOCK_N)

    # ---- Load L scale as (BLOCK_N, RANK//32) = (BLOCK_N, 2) ----
    offs_rk = tl.arange(0, RANK_GROUPS)
    l_scale = tl.load(
        L_scale_ptr + offs_n[:, None] * stride_lsn + offs_rk[None, :] * stride_lsk,
        mask=offs_n[:, None] < N,
        other=0,
    )   # (BLOCK_N, RANK_GROUPS)

    # ---- dot_scaled: (BLOCK_M, RANK) × (RANK, BLOCK_N) → (BLOCK_M, BLOCK_N) ----
    acc_lr = tl.dot_scaled(p_fp8, p_scale_e8m0, "e4m3",
                           l_tile, l_scale, "e4m3")

    # ===== Phase 3 — Sum + bias + channel-wise scaling + MXFP8 quant ========
    # (identical to Kernel 7)

    result = acc_lr + acc_q

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        result = result + bias[None, :]

    # ---- channel-wise pre-scaling before quantization ----
    cs = tl.load(
        channel_scale_ptr + offs_n * stride_cs,
        mask=offs_n < N, other=1.0,
    ).to(tl.float32)                                    # (BLOCK_N,)
    result = result * cs[None, :]                       # (BLOCK_M, BLOCK_N)

    N_GROUPS: tl.constexpr = BLOCK_N // SCALE_GROUP

    result_3d = tl.reshape(result, [BLOCK_M, N_GROUPS, SCALE_GROUP])
    amax = tl.max(tl.abs(result_3d), axis=2)

    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)

    scale_exp = tl.log2(amax).floor() - 7
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)
    scale_e8m0 = scale_exp.to(tl.uint8) + 127

    quant_scale = tl.exp2(-scale_exp)
    quant_scale_3d = tl.reshape(quant_scale, [BLOCK_M, N_GROUPS, 1])

    qx_3d = result_3d * quant_scale_3d
    qx = tl.reshape(qx_3d, [BLOCK_M, BLOCK_N])
    qx = tl.clamp(qx, min=-448.0, max=448.0)
    
    fp8_packed = qx.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_fp8_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        fp8_packed, mask=c_mask,
    )

    offs_ng = pid_n * N_GROUPS + tl.arange(0, N_GROUPS)
    s_mask = (offs_m[:, None] < M) & (offs_ng[None, :] < (N // SCALE_GROUP))
    tl.store(
        C_scale_ptr + offs_m[:, None] * stride_csm + offs_ng[None, :] * stride_csn,
        scale_e8m0, mask=s_mask,
    )

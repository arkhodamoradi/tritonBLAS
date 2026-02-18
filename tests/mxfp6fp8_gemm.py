
import triton
import triton.language as tl
import torch
import tcast
import numpy as np
import os
import triton.testing as tt

@triton.jit
def mxfp8_dot_scaled_gemm(
    A_ptr, B_ptr, C_ptr,
    As_ptr, Bs_ptr,                       # e8m0 scales (uint8)
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,

    # A: [M, K]
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # B: [N, K]
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    # C: [M, N]
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,

    # As: [M, K//32]
    stride_asm: tl.constexpr, stride_askg: tl.constexpr,
    # Bs: [N, K//32]  
    stride_bsn: tl.constexpr, stride_bskg: tl.constexpr,

    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    A_FMT: tl.constexpr = "e4m3",
    B_FMT: tl.constexpr = "e4m3",
    OUT_DTYPE: tl.constexpr = tl.float16, # tl.float16 / tl.bfloat16 / tl.float32
):
    # dot_scaled for MX formats uses group_size=32 for e8m0 scales
    GROUP_SIZE: tl.constexpr = 32
    tl.static_assert(BLOCK_K % GROUP_SIZE == 0, "BLOCK_K must be multiple of 32 for e8m0 scales")
    tl.static_assert(K % GROUP_SIZE == 0, "K must be multiple of 32 for e8m0 scales (MXFP8)")

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # simple grouped swizzle on M for L2 locality
    group_id = pid // (GROUP_M * grid_n)
    first_m = group_id * GROUP_M
    pid_in_group = pid % (GROUP_M * grid_n)
    pid_m = first_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_kg_tile = tl.arange(0, BLOCK_K // GROUP_SIZE)

    #tl.max_contiguous(offs_kg_tile, 16)
    #tl.multiple_of(offs_kg_tile, 16)

    # iterate over K tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_kg = (k0 // GROUP_SIZE) + offs_kg_tile
        
        #tl.multiple_of(k0, 512)

        # ----- load FP8 tiles -----
        # A: [BM, BK]
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # B: [BN, BK]
        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # ----- load e8m0 scales (uint8) -----
        # As: [BM, BK/32]
        a_scale = tl.load(
            As_ptr + offs_m[:, None] * stride_asm + offs_kg[None, :] * stride_askg,
            mask=(offs_m[:, None] < M) & (offs_kg[None, :] < (K // GROUP_SIZE)),
            other=0,
        )

        # Bs: [BN, BK/32]  (IMPORTANT: Bs is indexed by N then K-group; do NOT transpose)
        b_scale = tl.load(
            Bs_ptr + offs_n[:, None] * stride_bsn + offs_kg[None, :] * stride_bskg,
            mask=(offs_n[:, None] < N) & (offs_kg[None, :] < (K // GROUP_SIZE)),
            other=0,
        )

        # ----- scaled dot (native MX path when supported) -----
        acc = tl.dot_scaled(
            a, a_scale, A_FMT,
            b, b_scale, B_FMT,
            acc=acc,
            out_dtype=tl.float32,
        )

    # store
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(OUT_DTYPE),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

def mxfp6fp8_gemm(_A, _B, As, Bs,
              BM=128, BN=128, BK=128,
              num_warps=8, group_m=8, num_stages=1,
              _afmt="e4m3", _bfmt="e4m3", out_dtype=torch.float16):

    if _afmt == "e2m3":
        # unpack FP6 format to FP8 for GEMM
        A = triton_fused_unpack_gemm_kernel(_A[0], _A[1])
        afmt = "e4m3"
    else:
        A = _A
        afmt = _afmt
    if _bfmt == "e2m3":
        # unpack FP6 format to FP8 for GEMM
        B = triton_fused_unpack_gemm_kernel(_B[0], _B[1])
        bfmt = "e4m3"
    else:
        B = _B
        bfmt = _bfmt

    M, K = A.shape
    N, K2 = B.shape
    assert K == K2
    assert As.shape == (M, K//32)
    assert Bs.shape == (N, K//32)
    assert As.dtype == torch.uint8 and Bs.dtype == torch.uint8

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    As = As.contiguous()
    Bs = Bs.contiguous()
    
    assert As.stride(1) == 1   # stride_askg
    assert Bs.stride(1) == 1   # stride_bskg

    ms = tt.do_bench( lambda: mxfp8_dot_scaled_gemm[grid](
        A, B, C, As, Bs,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        As.stride(0), As.stride(1),
        Bs.stride(0), Bs.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_M=group_m,
        A_FMT=afmt, B_FMT=bfmt,
        OUT_DTYPE=tl.float16 if out_dtype == torch.float16 else (tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    ), warmup=10, rep=1000)

    print(f"Profiler: {ms:.4f}")

    return C

@triton.jit
def triton_fp6_pack_24bit_kernel(
    fp6_packed_ptr, output_ptr,
    n_elements, n_groups,
    BLOCK_SIZE: tl.constexpr
):
    """Pack 4 FP6 values into 24 bits (Mode 1) with automatic padding"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Each group processes 4 FP6 values
    base_idx = pid * 4
    
    # Initialize packed value as scalar with explicit type
    packed_24bit = tl.zeros((), dtype=tl.uint32)
    
    # Process each of the 4 FP6 values (already converted to 6-bit format)
    for i in tl.static_range(4):
        idx = base_idx + i
        # Load FP6 value if within bounds, otherwise use 0 (padding)
        if idx < n_elements:
            fp6_val = tl.load(fp6_packed_ptr + idx)
        else:
            fp6_val = tl.zeros((), dtype=tl.uint8)
            
        # Add to packed value at correct position
        shift = i * 6
        packed_24bit = packed_24bit | (fp6_val.to(tl.uint32) << shift)
    
    # Write 3 bytes - convert to proper type first
    packed_uint = packed_24bit.to(tl.uint32)
    byte0 = (packed_uint & 0xFF).to(tl.uint8)
    byte1 = ((packed_uint >> 8) & 0xFF).to(tl.uint8)
    byte2 = ((packed_uint >> 16) & 0xFF).to(tl.uint8)
    
    out_idx = pid * 3
    tl.store(output_ptr + out_idx, byte0)
    tl.store(output_ptr + out_idx + 1, byte1)
    tl.store(output_ptr + out_idx + 2, byte2)

@triton.jit
def triton_descale_and_pack_kernel(
    fp6_tensor_ptr, scale_tensor_ptr, output_ptr,
    N, n_scales, group_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Dequantize FP6 fake tensor and pack into 8-bit format for mode 0.
    
    Args:
        fp6_tensor_ptr: Pointer to FP6 tensor values
        scale_tensor_ptr: Pointer to scale tensor (int32 exponents)
        output_ptr: Pointer to output tensor (packed FP6 in uint8)
        N: Total number of elements
        n_scales: Number of scale values
        group_size: Number of elements per scale group
        BLOCK_SIZE: Block size for kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load FP6 values (already in bfloat16 format from tcast)
    vals = tl.load(fp6_tensor_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)

    # Compute which scale group each element belongs to
    scale_indices = offsets // group_size
    scale_mask = scale_indices < n_scales
    
    # Load scale values (int32 exponents)
    scale_vals = tl.load(scale_tensor_ptr + scale_indices, mask=mask & scale_mask, other=0).to(tl.uint8)

    # Compute scaling factor: 2^(127 - S)
    scale_factors = tl.exp2((127 - scale_vals).to(tl.float32)).to(tl.bfloat16)

    # Apply scale
    scaled_vals = vals * scale_factors
      
    # Convert to int16 for bit manipulation
    bf16_int = scaled_vals.to(tl.int16, bitcast=True)
    
    # Extract components from BF16: [sign(1)][exponent(8)][mantissa(7)]
    sign = (bf16_int >> 15) & 0x1
    exponent = (bf16_int >> 7) & 0xFF
    mantissa = bf16_int & 0x7F
    
    # Convert to FP6 format
    fp6_exponent = tl.where((exponent < 127), 0, exponent - 126)
    #fp6_mantissa = mantissa >> 4  # Take top 3 bits of mantissa
    fp6_mantissa =  tl.where((exponent == 124), 1, 
                    tl.where((exponent == 125), 2 | (mantissa >> 6), 
                    tl.where((exponent == 126), 4 | (mantissa >> 5), mantissa >> 4)))  # Handle subnormals

    # Pack into FP6 format: [00][sign(1)][exp(2)][mantissa(3)]
    fp6_packed = (sign << 5) | (fp6_exponent << 3) | fp6_mantissa
    
    # For both modes, store the FP6 values as uint8
    # Mode 1 will require a separate kernel to pack into 24-bit groups
    tl.store(output_ptr + offsets, fp6_packed.to(tl.uint8), mask=mask)

def triton_fused_pack_kernel(tcast_tensor):
    fp6_tensor = tcast_tensor.tensor
    scale_tensor = tcast_tensor.scaledata.scale
    n_elements = fp6_tensor.numel()
    n_scales = scale_tensor.numel()
    
    # Determine group size (elements per scale)
    group_size = n_elements // n_scales if n_scales > 0 else n_elements
    
    # Allocate output tensor for packed values (uint8)
    device = fp6_tensor.device
    fp6_output = torch.empty(n_elements, dtype=torch.uint8, device=device)

    # Launch dedescale and packing kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_descale_and_pack_kernel[grid](
        fp6_tensor, scale_tensor, fp6_output,
        n_elements, n_scales, group_size,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Only Mode 1: 24-bit packing (4 FP6 values -> 3 bytes)
    # Calculate groups using ceiling division - kernel handles padding automatically
    n_groups = triton.cdiv(n_elements, 4)
    n_bytes = n_groups * 3
    
    # Allocate output for 24-bit packed values
    packed_output = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    
    # Launch 24-bit packing kernel
    grid = (n_groups,)
    BLOCK_SIZE = 1
    
    triton_fp6_pack_24bit_kernel[grid](
        fp6_output, packed_output,
        n_elements, n_groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to (-1, 3) like CPU implementation
    return packed_output.reshape(-1, 3)

@triton.jit
def triton_fp6_unpack_24bit_kernel(
    packed_ptr, fp8_ptr,
    n_groups, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Unpack 24-bit packed format to FP8 E4M3 and BF16 values (Mode 1)"""
    pid = tl.program_id(0)
    
    if pid >= n_groups:
        return
    
    # Load 3 bytes for this group
    byte_idx = pid * 3
    byte0 = tl.load(packed_ptr + byte_idx).to(tl.uint32)
    byte1 = tl.load(packed_ptr + byte_idx + 1).to(tl.uint32)
    byte2 = tl.load(packed_ptr + byte_idx + 2).to(tl.uint32)
    
    # Reconstruct 24-bit value
    packed_24bit = byte0 | (byte1 << 8) | (byte2 << 16)
    
    # Extract 4 FP6 values
    base_idx = pid * 4
    for i in tl.static_range(4):
        idx = base_idx + i
        if idx < n_elements:
            # Extract 6-bit FP6 value
            shift = i * 6
            fp6_val = (packed_24bit >> shift) & 0x3F
            
            # Extract FP6 components: [S][EE][MMM]
            sign = ((fp6_val >> 5) & 0x1).to(tl.int32)
            fp6_exponent = ((fp6_val >> 3) & 0x3).to(tl.int32)
            fp6_mantissa = (fp6_val & 0x7).to(tl.int32)
            
            # 1. Convert to FP8 format
            fp8_exponent = tl.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0, 
                            tl.where((fp6_exponent == 0) & (fp6_mantissa == 1), 4,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), 5,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 3), 6, fp6_exponent + 6))))

            fp8_mantissa = tl.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0, 
                            tl.where((fp6_exponent == 0) & (fp6_mantissa == 1), 0,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), (fp6_mantissa & 1) << 2,
                            tl.where((fp6_exponent == 0) & (fp6_mantissa > 3), (fp6_mantissa & 3) << 1, fp6_mantissa))))
            
            fp8_packed = (sign << 7) | (fp8_exponent << 3) | fp8_mantissa
            tl.store(fp8_ptr + idx, fp8_packed.to(tl.uint8))

def triton_fused_unpack_gemm_kernel(packed_tensor, original_shape=None):
    device = packed_tensor.device
    
    if device.type != 'cuda':
        raise ValueError("Triton implementation requires CUDA device")

    # Only Mode 1: 24-bit packing (3 bytes contain 4 FP6 values)
    packed_bytes = packed_tensor.flatten()
    n_groups = packed_bytes.numel() // 3
    n_elements = n_groups * 4
    
    # Allocate output tensors
    fp8_output = torch.empty(n_elements, dtype=torch.uint8, device=device)
    
    # Launch kernel
    grid = (n_groups,)
    BLOCK_SIZE = 1
    triton_fp6_unpack_24bit_kernel[grid](
        packed_bytes, fp8_output,
        n_groups, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Trim to original size if shape provided
    fp8_output = fp8_output.view(torch.float8_e4m3fn)
    if original_shape:
        original_size = np.prod(original_shape)
        fp8_output = fp8_output[:original_size]
        return fp8_output.reshape(original_shape)

    return fp8_output    

CASTDICT = {
    "e4m3": tcast.mxfp8e4,
    "e2m3": tcast.mxfp6e2,
    "e2m1": tcast.mxfp4e2,
}
def get_scale_element_tcast(x, xfmt):
    tcast_dtype = CASTDICT.get(xfmt)
    x_tcast = tcast.cast(x, tcast_dtype)
    M, N = x.shape
    if xfmt == "e4m3":
        x_s = 2**((x_tcast.scaledata.scale - 127)).to(torch.float32).T
        x_q = (x_tcast.tensor.view(M, -1, 32) / x_s.view(M, -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(M, N)
        x_scale = x_tcast.scaledata.scale.to(torch.uint8).T.reshape(M, -1)
    elif xfmt == "e2m3":
        x_scale = x_tcast.scaledata.scale.to(torch.uint8).T.reshape(M, -1)
        x_q = (triton_fused_pack_kernel(x_tcast), x_tcast.tensor.shape)
    else:
        raise NotImplementedError(f"Format {xfmt} not implemented")

    return x_q, x_scale, x_tcast

def test_mxfp6fp8_dot_scaled_gemm(A, B, afmt, bfmt, bm, bn, bk, group_m, num_warps, num_stages):
    
    assert A.shape[1] == B.shape[1], "Inner dimensions must match for GEMM"
    M, N, K = A.shape[0], B.shape[0], A.shape[1]

    A_q, A_scale, A_tcast = get_scale_element_tcast(A, afmt)
    B_q, B_scale, B_tcast = get_scale_element_tcast(B, bfmt)

    C = mxfp6fp8_gemm(A_q, B_q, A_scale, B_scale, _afmt=afmt, _bfmt=bfmt, BM=bm, BN=bn, BK=bk, group_m=group_m, num_warps=num_warps, num_stages=num_stages, out_dtype=torch.float32)
    C_torch = A_tcast.tensor @ B_tcast.tensor.T

    print(f"Triton vs Torch error for ({afmt}, {bfmt}): {torch.max(torch.abs(C - C_torch))}")

def search_in_files(root_dir, filename_pattern=None, search_strings=["buffer_load_ubyte"], print_lines=False):
    out_dict = {}
    for search_string in search_strings:
        out_dict[search_string] = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if filename_pattern and not file.endswith(filename_pattern):
                continue

            file_path = os.path.join(root, file)
            # copy the file to the current directory
            os.system(f"cp {file_path} .")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        for search_string in search_strings:
                            if search_string in line:
                                if print_lines:
                                    print(f"{file_path}:{lineno}: {line.strip()}")
                                out_dict[search_string] += 1
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
    for search_string, count in out_dict.items():
        print(f"Total occurrences of '{search_string}': {count}")

if __name__ == "__main__":
    M = N = 64
    BM = BN = 64
    K = 128
    BK = 128
    GROUP_M = 1
    NUM_WARPS = 1
    NUM_STAGES = 1
    AFMTS = ["e4m3"] #, "e2m3"]
    BFMTS = ["e4m3"] #, "e2m3"]
    STRINGS = ["buffer_load_ubyte", "buffer_load_ushort", "buffer_load_dword ", "buffer_load_dwordx2"]

    A = torch.randn((M, K), device="cuda")
    B = torch.randn((N, K), device="cuda")
    
    for afmt in AFMTS:
        for bfmt in BFMTS:
            test_mxfp6fp8_dot_scaled_gemm(A, B, afmt=afmt, bfmt=bfmt, bm=BM, bn=BN, bk=BK, group_m=GROUP_M, num_warps=NUM_WARPS, num_stages=NUM_STAGES)

    search_in_files("/root/.triton/cache/", filename_pattern="mxfp8_dot_scaled_gemm.amdgcn", search_strings=STRINGS)

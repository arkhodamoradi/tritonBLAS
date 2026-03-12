# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triangle multiplicative update layers. Includes TriangleMultiplicativeUpdate from AF2
and FusedTriangleMultiplicativeUpdate from AF2-Multimer.
"""

import importlib
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partialmethod
import time

import torch
import torch.nn as nn
import torch.profiler

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.tensor_utils import permute_final_dims

cueq_is_installed = importlib.util.find_spec("cuequivariance_torch") is not None
if cueq_is_installed:
    from cuequivariance_torch import triangle_multiplicative_update

import triton
import triton.language as tl

warnings.filterwarnings("once")

# ============================================================================
# Profiling Infrastructure for Triton Kernels
# ============================================================================

# Global profiling flag - set to True to enable profiling
_PROFILING_ENABLED = False

# Kernel timing storage for detailed per-kernel statistics
_KERNEL_TIMINGS = {}

def enable_profiling():
    """Enable profiling for all Triton kernel calls."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = True

def disable_profiling():
    """Disable profiling for all Triton kernel calls."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = False

def get_kernel_timings():
    """Get accumulated kernel timing statistics."""
    return _KERNEL_TIMINGS.copy()

def reset_kernel_timings():
    """Reset all kernel timing statistics."""
    global _KERNEL_TIMINGS
    _KERNEL_TIMINGS = {}

def _record_kernel_time(kernel_name: str, elapsed_ms: float):
    """Record timing for a kernel execution."""
    global _KERNEL_TIMINGS
    if kernel_name not in _KERNEL_TIMINGS:
        _KERNEL_TIMINGS[kernel_name] = {
            'count': 0,
            'total_ms': 0.0,
            'min_ms': float('inf'),
            'max_ms': 0.0,
        }
    stats = _KERNEL_TIMINGS[kernel_name]
    stats['count'] += 1
    stats['total_ms'] += elapsed_ms
    stats['min_ms'] = min(stats['min_ms'], elapsed_ms)
    stats['max_ms'] = max(stats['max_ms'], elapsed_ms)

@contextmanager
def triton_kernel_profiler(name: str):
    """
    Context manager for profiling Triton kernel calls.
    
    When profiling is enabled, this:
    1. Records the kernel name in torch.profiler for trace visualization
    2. Measures GPU execution time using CUDA events
    3. Accumulates statistics in _KERNEL_TIMINGS
    
    Usage:
        with triton_kernel_profiler("batched_gemm_multi_channel"):
            batched_gemm_multi_channel[grid](...)
    """
    if _PROFILING_ENABLED:
        # Use torch.profiler.record_function for trace visualization
        with torch.profiler.record_function(f"triton::{name}"):
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            yield
            end_event.record()
            
            # Synchronize to get accurate timing
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            _record_kernel_time(name, elapsed_ms)
    else:
        yield

@contextmanager
def profile_triangle_update(output_dir: str = "./profile_results", export_trace: bool = True):
    """
    Context manager to profile all Triton kernels in triangle multiplicative update.
    
    This provides:
    1. Per-kernel timing statistics (printed as a table)
    2. Chrome trace export for visualization in chrome://tracing
    3. Memory profiling
    
    Usage:
        from openfold3.core.model.layers.triangular_multiplicative_update import (
            TriangleMultiplicativeUpdate,
            profile_triangle_update
        )
        
        model = TriangleMultiplicativeUpdate(c_z=128, c_hidden=32).cuda()
        z = torch.randn(1, 256, 256, 128, device='cuda', dtype=torch.bfloat16)
        mask = torch.ones(1, 256, 256, device='cuda', dtype=torch.bfloat16)
        
        with profile_triangle_update("./profiles") as prof:
            for _ in range(10):  # Warmup + profiling iterations
                output = model(z, mask)
        
        # This will print a table like:
        # Kernel Name                                    Calls   Total(ms)   Avg(ms)   Min(ms)   Max(ms)
        # batched_gemm_multi_channel                     10      23.45       2.35      2.10      2.80
        # fused_layernorm_gemm_sigmoid_multiply_a        10      18.20       1.82      1.70      2.00
        # fused_layernorm_gemm_sigmoid_multiply_b        10      17.80       1.78      1.65      1.95
    
    Args:
        output_dir: Directory to save profiling results
        export_trace: Whether to export Chrome trace JSON
    
    Yields:
        torch.profiler.profile object for additional analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable our custom profiling
    enable_profiling()
    reset_kernel_timings()
    
    # Configure torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof
    
    # Disable profiling
    disable_profiling()
    
    # Export Chrome trace if requested
    if export_trace:
        trace_path = os.path.join(output_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to: {trace_path}")
        print("Open chrome://tracing in Chrome and load this file to visualize.")
    
    # Print kernel timing statistics
    timings = get_kernel_timings()
    if timings:
        print("\n" + "=" * 100)
        print("TRITON KERNEL TIMING STATISTICS")
        print("=" * 100)
        print(f"{'Kernel Name':<55} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-" * 100)
        
        # Sort by total time descending
        sorted_kernels = sorted(timings.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        for kernel_name, stats in sorted_kernels:
            avg_ms = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0
            print(f"{kernel_name:<55} {stats['count']:>8} {stats['total_ms']:>12.3f} {avg_ms:>10.3f} {stats['min_ms']:>10.3f} {stats['max_ms']:>10.3f}")
        
        print("-" * 100)
        total_time = sum(s['total_ms'] for s in timings.values())
        total_calls = sum(s['count'] for s in timings.values())
        print(f"{'TOTAL':<55} {total_calls:>8} {total_time:>12.3f}")
        print("=" * 100)
    
    # Print torch.profiler summary
    print("\n" + "=" * 100)
    print("TORCH PROFILER SUMMARY (Top 20 by CUDA time)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

def print_kernel_summary():
    """Print a summary of kernel timings collected so far."""
    timings = get_kernel_timings()
    if not timings:
        print("No kernel timings recorded. Enable profiling with enable_profiling() first.")
        return
    
    print("\n" + "=" * 80)
    print("TRITON KERNEL TIMING SUMMARY")
    print("=" * 80)
    print(f"{'Kernel Name':<45} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10}")
    print("-" * 80)
    
    sorted_kernels = sorted(timings.items(), key=lambda x: x[1]['total_ms'], reverse=True)
    for kernel_name, stats in sorted_kernels:
        avg_ms = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{kernel_name:<45} {stats['count']:>8} {stats['total_ms']:>12.3f} {avg_ms:>10.3f}")
    
    print("=" * 80)

# ============================================================================
# Triton Kernels
# ============================================================================

# Multi-channel batched GEMM kernel that processes multiple channels per launch
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=2),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def batched_gemm_multi_channel(
    A_ptr, B_ptr, C_ptr,
    stride_Ab, stride_Ai, stride_Ak,
    stride_Bb, stride_Bk, stride_Bj,
    stride_Cb, stride_Ci, stride_Cj,
    M, K, N,
    total_channels,
    channels_per_batch,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # batch index 
    batch_group = tl.program_id(0)
    
    # tile indices
    pid = tl.program_id(1)  # Linear tile index
    
    # Number of tiles
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzling calculation
    num_pid_in_group = GROUP_SIZE_M * num_tiles_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    
    # Final tile indices
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    # Rest of kernel uses pid_m and pid_n...
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Process multiple channels in this kernel
    start_channel = batch_group * channels_per_batch
    end_channel = tl.minimum(start_channel + channels_per_batch, total_channels)
    
    # Process each channel in the group
    for channel in range(start_channel, end_channel):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            
            A = tl.load(
                A_ptr + channel * stride_Ab
                      + offs_m[:, None] * stride_Ai
                      + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            
            B = tl.load(
                B_ptr + channel * stride_Bb
                      + offs_k[:, None] * stride_Bk
                      + offs_n[None, :] * stride_Bj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            
            acc = tl.dot(A, B, acc)
        acc_bf16 = acc.to(tl.bfloat16)
        tl.store(
            C_ptr + channel * stride_Cb
                  + offs_m[:, None] * stride_Ci
                  + offs_n[None, :] * stride_Cj,
            acc_bf16,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

# Fused LayerNorm + GEMM + sigmoid + multiply kernel with channel-major output
# This kernel fuses the layer norm computation into the GEMM for the input
# Uses Welford's algorithm for single-pass mean/variance computation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=2),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def fused_layernorm_gemm_sigmoid_multiply_channel_major(
    A_ptr, W_gp_ptr,  # Input and CONCATENATED weight matrix [K, 2*N] for gate+proj
    C_ptr,
    bias_gp_ptr,  # CONCATENATED bias [2*N] for gate+proj (or None)
    mask_ptr,
    # LayerNorm parameters for A
    ln_weight_ptr, ln_bias_ptr,
    stride_Ab, stride_Ai, stride_Ak,
    stride_Wk, stride_Wj,  # Strides for concatenated weight
    stride_Cb, stride_Ci, stride_Cj,
    M, K, N,  # N is c_hidden (output channels), K is c_z (input channels)
    B, N_spatial,  # B is number of batches, N_spatial is spatial dimension
    eps,  # epsilon for layer norm
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that computes with CONCATENATED weights (1 GEMM instead of 2):
    1. A_normed = LayerNorm(A)  (fused into the GEMM using Welford's algorithm)
    2. gp = A_normed @ W_gp.T + bias_gp  (single GEMM, W_gp = [W_g; W_p] concatenated)
    3. gate = gp[:, :N], proj = gp[:, N:]
    4. output = sigmoid(gate) * proj * mask
    
    Key optimization: Uses concatenated weights [K, 2*N] to do 1 GEMM instead of 2.
    Output is written in channel-major format [B*C, N, N].
    
    Grid layout: (num_m_tiles, num_n_tiles, 1)
    - program_id(0): row tile index
    - program_id(1): channel tile index (for output N channels)
    """
    pid_m = tl.program_id(0)  # Row tile index
    pid_n = tl.program_id(1)  # Channel tile index (output channels)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator for concatenated output [BLOCK_M, 2*BLOCK_N]
    # We compute gate and proj in a single GEMM by loading both weight columns
    # Use FP32 accumulator for numerical stability
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Welford's algorithm state for layer norm (FP32 for numerical stability)
    mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    M2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    count = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # First pass: compute mean and variance using Welford's algorithm
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load as native dtype (bf16), convert to FP32 only for stats
        A_raw = tl.load(
            A_ptr + offs_m[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        A = A_raw.to(tl.float32)
        
        valid_mask = offs_k[None, :] < K
        n_valid = tl.sum(valid_mask.to(tl.float32), axis=1)
        
        local_sum = tl.sum(tl.where(valid_mask, A, 0.0), axis=1)
        local_mean = tl.where(n_valid > 0, local_sum / n_valid, 0.0)
        
        local_diff = tl.where(valid_mask, A - local_mean[:, None], 0.0)
        local_m2 = tl.sum(local_diff * local_diff, axis=1)
        
        new_count = count + n_valid
        delta = local_mean - mean
        
        mean = tl.where(new_count > 0, mean + delta * n_valid / new_count, mean)
        M2 = M2 + local_m2 + delta * delta * count * n_valid / tl.where(new_count > 0, new_count, 1.0)
        
        count = new_count
    
    # Final variance and rstd
    var = tl.where(count > 0, M2 / count, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: compute GEMM with fused layer norm
    # Load both gate and proj weights from concatenated matrix in single pass
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load A tile as native dtype (bf16)
        A_raw = tl.load(
            A_ptr + offs_m[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        
        # Load layer norm weight and bias as native dtype
        ln_w = tl.load(ln_weight_ptr + offs_k, mask=offs_k < K, other=1.0)
        ln_b = tl.load(ln_bias_ptr + offs_k, mask=offs_k < K, other=0.0)
        
        # Load W_g tile (first N columns of concatenated weight)
        W_g = tl.load(
            W_gp_ptr + offs_k[:, None] * stride_Wk + offs_n[None, :] * stride_Wj,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        
        # Load W_p tile (second N columns of concatenated weight, offset by N)
        W_p = tl.load(
            W_gp_ptr + offs_k[:, None] * stride_Wk + (N + offs_n[None, :]) * stride_Wj,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        
        # Apply layer norm: (x - mean) * rstd * weight + bias
        # Convert to FP32 for normalization, then convert to weight dtype for dot product
        A_normed_fp32 = (A_raw.to(tl.float32) - mean[:, None]) * rstd[:, None] * ln_w.to(tl.float32)[None, :] + ln_b.to(tl.float32)[None, :]
        # Convert A_normed to match weight dtype (handles both bf16 and fp32 weights)
        A_normed = A_normed_fp32.to(W_g.dtype)
        
        # Accumulate matrix multiplications - inputs match weight dtype, FP32 accumulator
        acc_g = tl.dot(A_normed, W_g.to(A_normed.dtype), acc_g)
        acc_p = tl.dot(A_normed, W_p.to(A_normed.dtype), acc_p)
    
    # Add biases if provided (from concatenated bias)
    if bias_gp_ptr is not None:
        bias_g = tl.load(bias_gp_ptr + offs_n, mask=offs_n < N, other=0.0)
        bias_p = tl.load(bias_gp_ptr + N + offs_n, mask=offs_n < N, other=0.0)
        acc_g = acc_g + bias_g[None, :]
        acc_p = acc_p + bias_p[None, :]
    
    # Apply sigmoid to gate
    gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))
    
    # Multiply sigmoid(gate) * proj
    result = gate_sigmoid * acc_p
    
    # Apply mask if provided 
    if mask_ptr is not None:
        mask_shared = tl.load(
            mask_ptr + offs_m,
            mask=offs_m < M,
            other=0.0
        ).to(tl.float32)
        
        result = result * mask_shared[:, None]
    
    # Store result in channel-major format
    batch_idx = offs_m // (N_spatial * N_spatial)
    remainder = offs_m % (N_spatial * N_spatial)
    spatial_i = remainder // N_spatial
    spatial_j = remainder % N_spatial
    
    out_offsets = (batch_idx[:, None] * N + offs_n[None, :]) * stride_Cb + \
                  spatial_i[:, None] * stride_Ci + \
                  spatial_j[:, None] * stride_Cj
    
    tl.store(C_ptr + out_offsets, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Fused LayerNorm + GEMM + sigmoid + multiply kernel
# This kernel fuses the layer norm computation into the GEMM for the second input (A_p)
# while keeping the first input (A) as pre-normalized
# Uses Welford's algorithm for single-pass mean/variance computation (only 2 reads of A_p)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=2),
    ],
    key=['M', 'K_g', 'K_p', 'N'],
)
@triton.jit
def fused_layernorm_gemm_sigmoid_multiply(
    A_ptr,  # Input for gate GEMM (already normalized z_normed)
    A_p_ptr,  # Input for proj GEMM (x_flat, needs layer norm)
    W_g_ptr, W_p_ptr,  # Weight matrices for gate and proj
    C_ptr,
    bias_g_ptr, bias_p_ptr,
    # LayerNorm parameters for A_p
    ln_weight_ptr, ln_bias_ptr,
    # Strides
    stride_Ai, stride_Ak,
    stride_Api, stride_Apk,
    stride_Wgk, stride_Wgj,
    stride_Wpk, stride_Wpj,
    stride_Ci, stride_Cj,
    M, K_g, K_p, N,  # K_g is c_z (gate input dim), K_p is c_hidden (proj input dim)
    eps,  # epsilon for layer norm
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. x_normed = LayerNorm(A_p)  (fused into the GEMM using Welford's algorithm)
    2. gate = A @ W_g.T + bias_g
    3. proj = x_normed @ W_p.T + bias_p
    4. output = sigmoid(gate) * proj
    
    This fuses the layer_norm_out into the final GEMM computation.
    Uses Welford's algorithm for single-pass mean/variance (only 2 reads of A_p total).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulators for both gate and proj
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # For each row in the tile, we need to compute layer norm statistics
    # Using Welford's algorithm for single-pass mean and variance computation
    # This is done per-row since layer norm normalizes across the K_p dimension
    
    # Welford's algorithm state: mean and M2 (sum of squared differences)
    mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    M2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    count = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # First pass: compute mean and variance using Welford's algorithm (single pass)
    for k_start in range(0, K_p, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        A_p = tl.load(
            A_p_ptr + offs_m[:, None] * stride_Api + offs_k[None, :] * stride_Apk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_p),
            other=0.0,
        ).to(tl.float32)
        
        # Welford's algorithm - process each element in the block
        valid_mask = offs_k[None, :] < K_p
        
        # For each valid element, update Welford's state
        # We process the block as a whole using parallel Welford's
        n_valid = tl.sum(valid_mask.to(tl.float32), axis=1)
        
        # Compute local statistics for this block
        local_sum = tl.sum(tl.where(valid_mask, A_p, 0.0), axis=1)
        local_mean = tl.where(n_valid > 0, local_sum / n_valid, 0.0)
        
        # Compute local M2 (sum of squared differences from local mean)
        local_diff = tl.where(valid_mask, A_p - local_mean[:, None], 0.0)
        local_m2 = tl.sum(local_diff * local_diff, axis=1)
        
        # Combine with running statistics using parallel Welford's formula
        new_count = count + n_valid
        delta = local_mean - mean
        
        # Update mean
        mean = tl.where(new_count > 0, mean + delta * n_valid / new_count, mean)
        
        # Update M2 using the parallel combination formula
        M2 = M2 + local_m2 + delta * delta * count * n_valid / tl.where(new_count > 0, new_count, 1.0)
        
        count = new_count
    
    # Final variance and rstd
    var = tl.where(count > 0, M2 / count, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Gate GEMM: A @ W_g (A is already normalized z_normed)
    for k_start in range(0, K_g, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load A tile for gate computation - keep as native dtype (bf16)
        A = tl.load(
            A_ptr + offs_m[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_g),
            other=0.0,
        )
        
        # Load W_g tile - keep as native dtype (bf16)
        W_g = tl.load(
            W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
            mask=(offs_k[:, None] < K_g) & (offs_n[None, :] < N),
            other=0.0,
        )
        
        # bf16 inputs with FP32 accumulator
        acc_g = tl.dot(A, W_g.to(A.dtype), acc_g)
    
    # Proj GEMM: LayerNorm(A_p) @ W_p (fused layer norm - second read of A_p)
    for k_start in range(0, K_p, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load A_p tile - keep as native dtype
        A_p_raw = tl.load(
            A_p_ptr + offs_m[:, None] * stride_Api + offs_k[None, :] * stride_Apk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_p),
            other=0.0,
        )
        
        # Load layer norm weight and bias - keep as native dtype
        ln_w = tl.load(ln_weight_ptr + offs_k, mask=offs_k < K_p, other=1.0)
        ln_b = tl.load(ln_bias_ptr + offs_k, mask=offs_k < K_p, other=0.0)
        
        # Load W_p tile
        W_p = tl.load(
            W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
            mask=(offs_k[:, None] < K_p) & (offs_n[None, :] < N),
            other=0.0,
        )
        
        # Apply layer norm: (x - mean) * rstd * weight + bias
        # Convert to FP32 for normalization, then convert to weight dtype for dot product
        A_p_normed_fp32 = (A_p_raw.to(tl.float32) - mean[:, None]) * rstd[:, None] * ln_w.to(tl.float32)[None, :] + ln_b.to(tl.float32)[None, :]
        # Convert A_p_normed to match weight dtype (handles both bf16 and fp32 weights)
        A_p_normed = A_p_normed_fp32.to(W_p.dtype)
        
        # Inputs match weight dtype, FP32 accumulator
        acc_p = tl.dot(A_p_normed, W_p.to(A_p_normed.dtype), acc_p)
    
    # Add biases if provided
    if bias_g_ptr is not None:
        bias_g = tl.load(bias_g_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_g = acc_g + bias_g[None, :]
    
    if bias_p_ptr is not None:
        bias_p = tl.load(bias_p_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_p = acc_p + bias_p[None, :]
    
    # Apply sigmoid to gate
    gate_sigmoid = tl.sigmoid(acc_g)
    
    # Multiply sigmoid(gate) * proj
    result = gate_sigmoid * acc_p
    
    # Store result
    tl.store(
        C_ptr + offs_m[:, None] * stride_Ci + offs_n[None, :] * stride_Cj,
        result,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

def _is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# ============================================================================
# Fused Projection + Matmul Kernel
# ============================================================================
# This kernel fuses:
# 1. LayerNorm + 2 GEMMs (gate, proj) + sigmoid + multiply for 'a' projection
# 2. LayerNorm + 2 GEMMs (gate, proj) + sigmoid + multiply for 'b' projection  
# 3. Batched matrix multiplication: a @ b
# All in a single kernel launch, eliminating intermediate memory traffic.
#
# Key design: Process one output element (batch, channel, i, j) per thread block.
# For each output, we iterate over k and compute projections on-the-fly.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_I': 64, 'BLOCK_J': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_I': 32, 'BLOCK_J': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_I': 64, 'BLOCK_J': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_I': 32, 'BLOCK_J': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['N', 'c_z', 'c_hidden'],
)
@triton.jit
def fused_projection_matmul_kernel(
    # Input
    z_ptr,  # [B*N*N, c_z] flattened input
    mask_ptr,  # [B*N*N] flattened mask
    # Concatenated weights: [c_z, 2*c_hidden] for gate and proj combined
    W_a_ptr,  # W_a = concat(W_ag, W_ap) along last dim: [c_z, 2*c_hidden]
    W_b_ptr,  # W_b = concat(W_bg, W_bp) along last dim: [c_z, 2*c_hidden]
    # LayerNorm parameters
    ln_weight_ptr,  # [c_z]
    ln_bias_ptr,    # [c_z]
    # Output
    output_ptr,  # [B*c_hidden, N, N] channel-major output
    # Strides
    stride_z_row, stride_z_col,
    stride_W_k, stride_W_n,
    stride_out_c, stride_out_i, stride_out_j,
    # Dimensions
    B, N, c_z, c_hidden,
    eps,
    # Direction flag
    outgoing: tl.constexpr,
    # Block sizes
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that computes the entire triangle multiplicative update projection + matmul.
    
    For each output tile (batch, channel, i_tile, j_tile):
    - Iterate over k in blocks
    - For each k block, compute a[i, k] and b[k, j] projections on-the-fly
    - Accumulate the matmul result
    
    Grid: (B * c_hidden, num_i_tiles, num_j_tiles)
    Each program computes one (batch, channel) pair for a tile of (i, j) positions.
    """
    # Program IDs
    pid_bc = tl.program_id(0)  # batch * c_hidden + channel
    pid_i = tl.program_id(1)   # i tile
    pid_j = tl.program_id(2)   # j tile
    
    # Decompose pid_bc into batch and channel
    batch_idx = pid_bc // c_hidden
    channel_idx = pid_bc % c_hidden
    
    # Compute offsets for this tile
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    # Initialize output accumulator for this tile
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    # Iterate over k (the reduction dimension for the matmul)
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # ===== Compute 'a' projection for positions (i, k) =====
        # a[i, k] = sigmoid(z_normed[i,k] @ W_ag[channel]) * (z_normed[i,k] @ W_ap[channel]) * mask[i,k]
        
        # Row indices in z_flat for 'a': batch * N * N + i * N + k
        z_a_row_idx = batch_idx * N * N + offs_i[:, None] * N + offs_k[None, :]  # [BLOCK_I, BLOCK_K]
        
        # Compute LayerNorm statistics for 'a' positions
        # First pass: compute mean
        sum_a = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            z_a_chunk = tl.load(
                z_ptr + z_a_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_i[:, None, None] < N) & (offs_k[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            sum_a += tl.sum(z_a_chunk, axis=2)
        mean_a = sum_a / c_z
        
        # Second pass: compute variance
        var_sum_a = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            z_a_chunk = tl.load(
                z_ptr + z_a_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_i[:, None, None] < N) & (offs_k[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            diff = z_a_chunk - mean_a[:, :, None]
            var_sum_a += tl.sum(diff * diff, axis=2)
        var_a = var_sum_a / c_z
        rstd_a = 1.0 / tl.sqrt(var_a + eps)
        
        # Compute 'a' projection: accumulate over c_z
        a_gate = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)
        a_proj = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)
        
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            
            # Load z chunk and apply layer norm
            z_a_chunk = tl.load(
                z_ptr + z_a_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_i[:, None, None] < N) & (offs_k[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            
            ln_w = tl.load(ln_weight_ptr + offs_cz, mask=offs_cz < c_z, other=1.0).to(tl.float32)
            ln_b = tl.load(ln_bias_ptr + offs_cz, mask=offs_cz < c_z, other=0.0).to(tl.float32)
            
            z_a_normed = (z_a_chunk - mean_a[:, :, None]) * rstd_a[:, :, None] * ln_w[None, None, :] + ln_b[None, None, :]
            
            # Load weights for this channel (gate and proj) - load as 1D vectors
            W_a_gate_chunk = tl.load(
                W_a_ptr + offs_cz * stride_W_k + channel_idx * stride_W_n,
                mask=offs_cz < c_z,
                other=0.0
            ).to(tl.float32)  # [32]
            
            W_a_proj_chunk = tl.load(
                W_a_ptr + offs_cz * stride_W_k + (c_hidden + channel_idx) * stride_W_n,
                mask=offs_cz < c_z,
                other=0.0
            ).to(tl.float32)  # [32]
            
            # Accumulate: z_a_normed[BLOCK_I, BLOCK_K, 32] @ W[32] -> [BLOCK_I, BLOCK_K]
            # Use einsum-like reduction: sum over the c_z dimension
            a_gate += tl.sum(z_a_normed * W_a_gate_chunk[None, None, :], axis=2)
            a_proj += tl.sum(z_a_normed * W_a_proj_chunk[None, None, :], axis=2)
        
        # Apply sigmoid and multiply
        a_gate = tl.sigmoid(a_gate)
        a = a_gate * a_proj
        
        # Apply mask
        mask_a = tl.load(
            mask_ptr + z_a_row_idx,
            mask=(offs_i[:, None] < N) & (offs_k[None, :] < N),
            other=0.0
        ).to(tl.float32)
        a = a * mask_a
        
        # ===== Compute 'b' projection for positions (k, j) or (j, k) =====
        # For outgoing: b uses z[batch, k, j, :]
        # For incoming: b uses z[batch, j, k, :]
        if outgoing:
            z_b_row_idx = batch_idx * N * N + offs_k[:, None] * N + offs_j[None, :]  # [BLOCK_K, BLOCK_J]
        else:
            z_b_row_idx = batch_idx * N * N + offs_j[None, :] * N + offs_k[:, None]  # [BLOCK_J, BLOCK_K]
            z_b_row_idx = tl.trans(z_b_row_idx)  # [BLOCK_K, BLOCK_J]
        
        # Compute LayerNorm statistics for 'b' positions
        sum_b = tl.zeros((BLOCK_K, BLOCK_J), dtype=tl.float32)
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            z_b_chunk = tl.load(
                z_ptr + z_b_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_k[:, None, None] < N) & (offs_j[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            sum_b += tl.sum(z_b_chunk, axis=2)
        mean_b = sum_b / c_z
        
        var_sum_b = tl.zeros((BLOCK_K, BLOCK_J), dtype=tl.float32)
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            z_b_chunk = tl.load(
                z_ptr + z_b_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_k[:, None, None] < N) & (offs_j[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            diff = z_b_chunk - mean_b[:, :, None]
            var_sum_b += tl.sum(diff * diff, axis=2)
        var_b = var_sum_b / c_z
        rstd_b = 1.0 / tl.sqrt(var_b + eps)
        
        # Compute 'b' projection
        b_gate = tl.zeros((BLOCK_K, BLOCK_J), dtype=tl.float32)
        b_proj = tl.zeros((BLOCK_K, BLOCK_J), dtype=tl.float32)
        
        for cz_start in range(0, c_z, 32):
            offs_cz = cz_start + tl.arange(0, 32)
            
            z_b_chunk = tl.load(
                z_ptr + z_b_row_idx[:, :, None] * stride_z_row + offs_cz[None, None, :] * stride_z_col,
                mask=(offs_k[:, None, None] < N) & (offs_j[None, :, None] < N) & (offs_cz[None, None, :] < c_z),
                other=0.0
            ).to(tl.float32)
            
            ln_w = tl.load(ln_weight_ptr + offs_cz, mask=offs_cz < c_z, other=1.0).to(tl.float32)
            ln_b = tl.load(ln_bias_ptr + offs_cz, mask=offs_cz < c_z, other=0.0).to(tl.float32)
            
            z_b_normed = (z_b_chunk - mean_b[:, :, None]) * rstd_b[:, :, None] * ln_w[None, None, :] + ln_b[None, None, :]
            
            # Load weights for this channel (gate and proj) - load as 1D vectors
            W_b_gate_chunk = tl.load(
                W_b_ptr + offs_cz * stride_W_k + channel_idx * stride_W_n,
                mask=offs_cz < c_z,
                other=0.0
            ).to(tl.float32)  # [32]
            
            W_b_proj_chunk = tl.load(
                W_b_ptr + offs_cz * stride_W_k + (c_hidden + channel_idx) * stride_W_n,
                mask=offs_cz < c_z,
                other=0.0
            ).to(tl.float32)  # [32]
            
            b_gate += tl.sum(z_b_normed * W_b_gate_chunk[None, None, :], axis=2)
            b_proj += tl.sum(z_b_normed * W_b_proj_chunk[None, None, :], axis=2)
        
        b_gate = tl.sigmoid(b_gate)
        b = b_gate * b_proj
        
        mask_b = tl.load(
            mask_ptr + z_b_row_idx,
            mask=(offs_k[:, None] < N) & (offs_j[None, :] < N),
            other=0.0
        ).to(tl.float32)
        b = b * mask_b
        
        # ===== Accumulate matmul: a[BLOCK_I, BLOCK_K] @ b[BLOCK_K, BLOCK_J] =====
        acc += tl.dot(a, b.to(a.dtype)).to(tl.float32)
    
    # Store output in channel-major format [B*c_hidden, N, N]
    out_offset = (batch_idx * c_hidden + channel_idx) * stride_out_c + \
                 offs_i[:, None] * stride_out_i + \
                 offs_j[None, :] * stride_out_j
    tl.store(
        output_ptr + out_offset,
        acc.to(tl.bfloat16),
        mask=(offs_i[:, None] < N) & (offs_j[None, :] < N)
    )


def fused_build_and_combine_projections(z, mask, layer_norm_in, linear_a_g, linear_a_p, 
                                         linear_b_g, linear_b_p, c_hidden, outgoing):
    """
    Fused version that combines _build_projections and triton_combine_projection.
    
    Uses concatenated weights to do 2 GEMMs instead of 4.
    
    Args:
        z: [B, N, N, c_z] input tensor
        mask: [B, N, N, 1] mask tensor
        layer_norm_in: LayerNorm module
        linear_a_g, linear_a_p: Linear modules for 'a' projection
        linear_b_g, linear_b_p: Linear modules for 'b' projection
        c_hidden: hidden dimension
        outgoing: direction flag
    
    Returns:
        x: [B*c_hidden, N, N] output in channel-major format
        z_flat: [B*N*N, c_z] flattened input for _post_projections
    """
    original_shape = z.shape
    *batch_dims, N, _, c_z = original_shape
    
    # Calculate batch size
    Bflat = 1
    for d in batch_dims:
        Bflat *= d
    
    # Flatten inputs
    z_flat = z.reshape(-1, c_z)  # [B*N*N, c_z]
    mask_flat = mask.reshape(-1)  # [B*N*N]
    
    # Concatenate weights: [c_z, 2*c_hidden]
    W_a = torch.cat([linear_a_g.weight.T, linear_a_p.weight.T], dim=1)  # [c_z, 2*c_hidden]
    W_b = torch.cat([linear_b_g.weight.T, linear_b_p.weight.T], dim=1)  # [c_z, 2*c_hidden]
    
    # Get layer norm parameters
    ln_weight = layer_norm_in.weight
    ln_bias = layer_norm_in.bias
    
    # Allocate output
    output = torch.empty(Bflat * c_hidden, N, N, device=z.device, dtype=z.dtype)
    
    # Grid: one program per (batch, channel, i_tile, j_tile)
    def grid(meta):
        num_i_tiles = triton.cdiv(N, meta['BLOCK_I'])
        num_j_tiles = triton.cdiv(N, meta['BLOCK_J'])
        return (Bflat * c_hidden, num_i_tiles, num_j_tiles)
    
    # Launch kernel
    with triton_kernel_profiler("fused_projection_matmul"):
        fused_projection_matmul_kernel[grid](
            z_flat, mask_flat,
            W_a, W_b,
            ln_weight, ln_bias,
            output,
            z_flat.stride(0), z_flat.stride(1),
            W_a.stride(0), W_a.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            Bflat, N, c_z, c_hidden,
            1e-5,  # eps
            outgoing,
        )
    
    return output, z_flat


def triton_combine_projection(a, b, outgoing = True, batch_dims=None, c_hidden=None):
    """
    Optimized version that accepts inputs already in [B*C, N, N] format.
    Uses Option B: Keep channel-major output [B*C, N, N] and pass directly to _post_projections.
    
    Args:
        a: [B*C, N, N] tensor (already in channel-major format)
        b: [B*C, N, N] tensor (already in channel-major format)
        outgoing: whether this is outgoing or incoming update
        batch_dims: original batch dimensions (for reshaping output)
        c_hidden: hidden channel dimension (for reshaping output)
    
    Returns:
        [B*C, N, N] tensor (channel-major, contiguous) - NO reshape, NO contiguous needed
    """
    # Get dimensions - tensors should already be on correct device
    BC, N, _ = a.shape
    
    # Preserve input dtype for output
    ab_matmul = torch.empty(BC, N, N, device=a.device, dtype=a.dtype)

    # Process ONE channel per thread block for maximum parallelism
    # This allows all channels to run in parallel across the GPU
    channels_per_batch = 1
    num_batches = BC
    
    def grid(meta):
        num_m_tiles = triton.cdiv(N, meta['BLOCK_M'])
        num_n_tiles = triton.cdiv(N, meta['BLOCK_N'])
        total_tiles = num_m_tiles * num_n_tiles 
        return (
            num_batches,  # Reduced number of batches
            total_tiles,
        )
    
    # Run the batched GEMM kernel with profiling
    with triton_kernel_profiler("batched_gemm_multi_channel"):
        batched_gemm_multi_channel[grid](
            a, b, ab_matmul,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            ab_matmul.stride(0), ab_matmul.stride(1), ab_matmul.stride(2),
            N, N, N,  # M, K, N for square matrices
            BC,  # Total number of channels
            channels_per_batch,  # Channels to process per kernel
            # **kernel_kwargs,
        )

    # Return in [B*C, N, N] format - NO reshape, NO contiguous
    # _post_projections will handle this format directly
    return ab_matmul

        
class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Common base class for TriangleMultiplicativeUpdate and
    FusedTriangleMultiplicativeUpdate.
    """

    @abstractmethod
    def __init__(
        self, c_z, c_hidden, _outgoing, linear_init_params=lin_init.tri_mul_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, **linear_init_params.linear_g)
        self.linear_z = Linear(self.c_hidden, self.c_z, **linear_init_params.linear_z)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _build_projections(self, z, mask):
        """
        Optimized version that outputs a and b directly in [B*C, N, N] format.
        Uses fused LayerNorm + GEMM + sigmoid + multiply kernel with CONCATENATED weights.
        This does 1 GEMM per projection instead of 2 (gate and proj combined).
        """
        # Tensors should already be on the correct device from the model
        original_shape = z.shape
        *batch_dims, N, _, C = original_shape
        
        # Calculate batch size
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        
        # Flatten to 2D: [batch*N*N, C]
        z_flat = z.reshape(-1, C)
        N_rows = z_flat.shape[0]  # Total number of rows after flattening

        c_z = original_shape[-1]
        c_hidden = self.c_hidden
        
        # Get layer norm parameters
        ln_weight = self.layer_norm_in.weight
        ln_bias = self.layer_norm_in.bias
        
        # Allocate output tensors in channel-major format [B*C, N, N]
        a_channel_major = torch.empty(Bflat * c_hidden, N, N, device=z.device, dtype=z.dtype)
        b_channel_major = torch.empty(Bflat * c_hidden, N, N, device=z.device, dtype=z.dtype)
        
        # Grid for fused computation - use 2D grid to avoid CUDA dimension limits
        # program_id(0) is the row tile (can be up to 2^31-1)
        # program_id(1) is the channel tile
        def grid(meta):
            return (
                triton.cdiv(z_flat.shape[0], meta['BLOCK_M']),  # Row tiles (large dimension)
                triton.cdiv(c_hidden, meta['BLOCK_N']),  # Channel tiles
            )
        
        # CONCATENATE weights: [c_z, 2*c_hidden] for gate+proj combined
        # This allows 1 GEMM instead of 2 per projection
        W_a_gp = torch.cat([self.linear_a_g.weight.T, self.linear_a_p.weight.T], dim=1)  # [c_z, 2*c_hidden]
        W_b_gp = torch.cat([self.linear_b_g.weight.T, self.linear_b_p.weight.T], dim=1)  # [c_z, 2*c_hidden]
        
        # Concatenate biases if they exist
        bias_a_gp = None
        if self.linear_a_g.bias is not None and self.linear_a_p.bias is not None:
            bias_a_gp = torch.cat([self.linear_a_g.bias, self.linear_a_p.bias])  # [2*c_hidden]
        
        bias_b_gp = None
        if self.linear_b_g.bias is not None and self.linear_b_p.bias is not None:
            bias_b_gp = torch.cat([self.linear_b_g.bias, self.linear_b_p.bias])  # [2*c_hidden]
        
        mask_flat = mask.reshape(-1)
        
        # Process a projection with fused LayerNorm + GEMM + channel-major output
        # Uses concatenated weights for 1 GEMM instead of 2
        with triton_kernel_profiler("fused_layernorm_gemm_sigmoid_multiply_channel_major_a"):
            fused_layernorm_gemm_sigmoid_multiply_channel_major[grid](
                z_flat, W_a_gp,  # Concatenated weight [c_z, 2*c_hidden]
                a_channel_major,
                bias_a_gp,  # Concatenated bias [2*c_hidden] or None
                mask_flat,
                ln_weight, ln_bias,  # LayerNorm parameters
                0, z_flat.stride(0), z_flat.stride(1),
                W_a_gp.stride(0), W_a_gp.stride(1),  # Strides for concatenated weight
                a_channel_major.stride(0), a_channel_major.stride(1), a_channel_major.stride(2),
                z_flat.shape[0], c_z, c_hidden,
                Bflat, N,  # B and N_spatial for channel-major output
                1e-5,  # epsilon for layer norm
            )
        
        # Process b projection with fused LayerNorm + GEMM + channel-major output
        with triton_kernel_profiler("fused_layernorm_gemm_sigmoid_multiply_channel_major_b"):
            fused_layernorm_gemm_sigmoid_multiply_channel_major[grid](
                z_flat, W_b_gp,  # Concatenated weight [c_z, 2*c_hidden]
                b_channel_major,
                bias_b_gp,  # Concatenated bias [2*c_hidden] or None
                mask_flat,
                ln_weight, ln_bias,  # LayerNorm parameters
                0, z_flat.stride(0), z_flat.stride(1),
                W_b_gp.stride(0), W_b_gp.stride(1),  # Strides for concatenated weight
                b_channel_major.stride(0), b_channel_major.stride(1), b_channel_major.stride(2),
                z_flat.shape[0], c_z, c_hidden,
                Bflat, N,  # B and N_spatial for channel-major output
                1e-5,  # epsilon for layer norm
            )
        
        # Apply direction-specific permutations
        if self._outgoing:
            # For outgoing: b needs transpose
            b_channel_major = b_channel_major.transpose(-1, -2)
        else:
            # For incoming: a needs transpose
            a_channel_major = a_channel_major.transpose(-1, -2)
        
        return z_flat, a_channel_major, b_channel_major

    def _post_projections(self, x_flat, z_normed_h, orig_shape):
        """
        Optimized version that accepts x_flat in [B*C, N, N] format (channel-major)
        and returns in [..., N, N, C].
        Uses fused LayerNorm + GEMM + sigmoid + multiply kernel to avoid separate layer norm pass.
        
        Option B: x_flat is in [B*C, N, N] format from batched_gemm_multi_channel.
        We reshape it to [B*N*N, C] for the fused kernel without needing .contiguous().
        """
        # Tensors should already be on the correct device
        z_normed = z_normed_h
        
        # Get dimensions from orig_shape
        *batch_dims, N, _, C = orig_shape
        
        # Calculate batch size
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        
        c_hidden = self.c_hidden
        c_z = self.c_z
        
        # x_flat is [B*C, N, N] from batched_gemm_multi_channel (channel-major)
        # We need to reshape to [B*N*N, C] for the fused kernel
        # 
        # Current layout: [B*C, N, N] with strides [N*N, N, 1]
        # Target layout: [B*N*N, C] with strides [C, 1]
        #
        # We can do this by:
        # 1. Reshape to [B, C, N, N]
        # 2. Permute to [B, N, N, C]
        # 3. Reshape to [B*N*N, C]
        #
        # The permute creates a non-contiguous view, but the fused kernel
        # can handle non-contiguous inputs via strides!
        
        # Reshape [B*C, N, N] -> [B, C, N, N]
        x_reshaped = x_flat.reshape(Bflat, c_hidden, N, N)
        
        # Permute to [B, N, N, C] - this is a view, not a copy
        x_permuted = x_reshaped.permute(0, 2, 3, 1)
        
        # Reshape to [B*N*N, C] - this is also a view since the last dim is contiguous
        x_flat = x_permuted.reshape(-1, c_hidden)
        
        # Calculate batch size
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        
        # x_flat is already in [B*N*N, c_hidden] format
        N_rows = x_flat.shape[0]

        # Get layer norm parameters
        ln_weight = self.layer_norm_out.weight
        ln_bias = self.layer_norm_out.bias

        # Use fused LayerNorm + GEMM + sigmoid + multiply kernel
        # This fuses layer_norm_out into the GEMM computation
        output_flat = torch.empty(N_rows, c_z, device=x_flat.device, dtype=x_flat.dtype)
        
        # Grid for fused computation
        def grid(meta):
            return (
                triton.cdiv(N_rows, meta['BLOCK_M']),
                triton.cdiv(c_z, meta['BLOCK_N']),
            )
        
        # Get weight and bias pointers - weights should already be on correct device
        W_g = self.linear_g.weight.T
        W_z = self.linear_z.weight.T
        
        bias_g = self.linear_g.bias if self.linear_g.bias is not None else None
        bias_z = self.linear_z.bias if self.linear_z.bias is not None else None
         
        # Process final projection with fused LayerNorm + GEMM kernel
        # This computes: sigmoid(z_normed @ W_g.T + bias_g) * (LayerNorm(x_flat) @ W_z.T + bias_z)
        with triton_kernel_profiler("fused_layernorm_gemm_sigmoid_multiply"):
            fused_layernorm_gemm_sigmoid_multiply[grid](
                z_normed,  # A for gate (already normalized)
                x_flat,    # A_p for proj (needs layer norm, fused into kernel)
                W_g, W_z,  # Weight matrices
                output_flat,
                bias_g, bias_z,
                ln_weight, ln_bias,  # LayerNorm parameters for x_flat
                z_normed.stride(0), z_normed.stride(1),  # A strides
                x_flat.stride(0), x_flat.stride(1),  # A_p strides
                W_g.stride(0), W_g.stride(1),
                W_z.stride(0), W_z.stride(1),
                output_flat.stride(0), output_flat.stride(1),
                N_rows, c_z, c_hidden, c_z,  # M, K_g, K_p, N
                1e-5,  # epsilon for layer norm
            )
        
        # Reshape back to original shape with c_z
        output = output_flat.reshape(*batch_dims, N, N, c_z)

        return output

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.einsum(
                    "...ij,...jk->...ik", a_chunk, b_chunk
                )

            p = a
        else:
            p = torch.einsum("...ij,...jk->...ik", a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass
        
class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithms 11 and 12 / AF3 Algorithms 12 and 13.
    """

    def __init__(
        self, c_z, c_hidden, _outgoing=True, linear_init_params=lin_init.tri_mul_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=_outgoing,
            linear_init_params=linear_init_params,
        )

        self.linear_a_p = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_a_p
        )
        self.linear_a_g = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_a_g
        )
        self.linear_b_p = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_b_p
        )
        self.linear_b_g = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_b_g
        )

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_chunk_size: int | None = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences.

        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the "square" input tensor, 2) a, the first projection of z,
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a
        z-sized tensor for intermediate computations. For large N, this is
        prohibitively expensive; for N=4000, for example, z is more than 8GB
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding
        vertical and horizontal chunks of z. This suggests an algorithm that
        loops over pairs of chunks of z: hereafter "columns" and "rows" of
        z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith "column" of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the
        ith column is always one column ahead of previously overwritten columns
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i
        exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th
        quadrants of z instead. Though the 3rd quadrant of the original z is
        entirely overwritten at this point, it can be recovered from the z-cache
        itself. Thereafter, the ith row of z can be recovered in its entirety
        from the reoriented z-cache. After the final iteration, z has been
        completely overwritten and contains the triangular multiplicative
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory
        consumption is just 2.5x the size of z, disregarding memory used for
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.weight.shape[-2]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[tuple(s)]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[tuple(first_half_slicer)] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[tuple(quadrant_3_slicer)] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[tuple(z_cache_slicer)])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [
                i_2 - i_1
                for i_1, i_2 in zip(i_range, i_range[1:] + [half_n], strict=True)
            ]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets, strict=False
            )
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[tuple(z_chunk_slicer)] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache, z_cache_offset, z_cache_offset + offset, row_dim
                        )

                b_chunk = compute_projection(
                    z_chunk_b, mask_chunk, a=False, chunked=False
                )
                del z_chunk_b

                x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[tuple(z_slicer)] += x_chunk
                else:
                    z[tuple(z_slicer)] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.einsum("...ij,...jk->...ik", a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        ## NOTE: valid for inplace safe and use_cueq_triangle_kernels to be enabled
        ## inplace safe is used across the codebase and so should not
        ## be disabled. So if use_cueq_triangle_kernels is True, it will always
        ## supersede inplace_safe
        # if use_cueq_triangle_kernels:
        #     ## VS: The cuequivariance kernel is based on the boltz implementation
        #     ## of triangle multiplicative update, which fuses the linear_*_p
        #     ## projections into a single layer (similarly for linear_*_g).
        #     ## this why we need to concat the projection layers here
        #     x = _cueq_triangle_mult(
        #         z=z,
        #         g_in_weight=torch.cat(
        #             [
        #                 self.linear_a_g.weight,
        #                 self.linear_b_g.weight,
        #             ]
        #         ),
        #         p_in_weight=torch.cat(
        #             [
        #                 self.linear_a_p.weight,
        #                 self.linear_b_p.weight,
        #             ]
        #         ),
        #         _outgoing=self._outgoing,
        #         mask=mask,
        #         norm_in_weight=self.layer_norm_in.weight,
        #         norm_in_bias=self.layer_norm_in.bias,
        #         norm_out_weight=self.layer_norm_out.weight,
        #         norm_out_bias=self.layer_norm_out.bias,
        #         p_out_weight=self.linear_z.weight,
        #         g_out_weight=self.linear_g.weight,
        #     )
        #     return x

        # if inplace_safe:
        #     x = self._inference_forward(
        #         z,
        #         mask,
        #         inplace_chunk_size=_inplace_chunk_size,
        #         with_add=_add_with_inplace,
        #     )
        #     return x

        # if mask is None:
        #     mask = z.new_ones(z.shape[:-1])

        # mask = mask.unsqueeze(-1)

        # if torch.cuda.is_available() and z.is_cuda:
            
        z_flat, a, b = self._build_projections(z, mask)
        x = triton_combine_projection(a, b, self._outgoing, batch_dims=z.shape[:-3], c_hidden=self.c_hidden)
        del a, b
        x = self._post_projections(x, z_flat, z.shape)
        return x
        # else:
        #     z = self.layer_norm_in(z)
        #     a = mask  # (1,s, s, 1)
        #     a = a * self.sigmoid(self.linear_a_g(z))
        #     a = a * self.linear_a_p(z)
        #     b = mask
        #     b = b * self.sigmoid(self.linear_b_g(z))
        #     b = b * self.linear_b_p(z)

        #     x = self._combine_projections(a, b)

        #     del a, b
        #     x = self.layer_norm_out(x)
        #     x = self.linear_z(x)
        #     g = self.sigmoid(self.linear_g(z))
        #     x = x * g

        #     return x

class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithm 11 / AF3 Algorithm 12.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithm 12 / AF3 Algorithm 13.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class FusedTriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 11 and 12.
    Not compatible with AF3 - Linear layers here are instantiated with
    biases, compared to AF3 version which uses LinearNoBias
    """

    def __init__(
        self,
        c_z,
        c_hidden,
        _outgoing=True,
        linear_init_params=lin_init.fused_tri_mul_init,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=_outgoing,
            linear_init_params=linear_init_params,
        )

        self.linear_ab_p = Linear(
            self.c_z, self.c_hidden * 2, **linear_init_params.linear_ab_p
        )
        self.linear_ab_g = Linear(
            self.c_z, self.c_hidden * 2, **linear_init_params.linear_ab_g
        )

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        _inplace_chunk_size: int | None = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask):
            p = self.linear_ab_g(pair)
            p.sigmoid_()
            p *= self.linear_ab_p(pair)
            p *= mask

            return p

        def compute_projection(pair, mask):
            p = compute_projection_helper(pair, mask)
            left = p[..., : self.c_hidden]
            right = p[..., self.c_hidden :]

            return left, right

        z_norm_in = self.layer_norm_in(z)
        a, b = compute_projection(z_norm_in, mask)
        x = self._combine_projections(a, b, _inplace_chunk_size=_inplace_chunk_size)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.linear_g(z_norm_in)
        g.sigmoid_()
        x *= g
        if with_add:
            z += x
        else:
            z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if use_cueq_triangle_kernels:
            raise NotImplementedError(
                "CUEQ triangle multiplicative update kernel not"
                "supported for FusedTriangleMultiplicativeUpdate."
                "\nPlease change config"
            )

        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                _inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        ab = mask
        ab = ab * self.sigmoid(self.linear_ab_g(z))
        ab = ab * self.linear_ab_p(z)

        a = ab[..., : self.c_hidden]
        b = ab[..., self.c_hidden :]

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class FusedTriangleMultiplicationOutgoing(FusedTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 11.
    Not compatible with AF3
    """

    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=True)


class FusedTriangleMultiplicationIncoming(FusedTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 12.
    Not compatible with AF3
    """

    __init__ = partialmethod(
        FusedTriangleMultiplicativeUpdate.__init__, _outgoing=False
    )


def _cueq_triangle_mult(
    z: torch.Tensor,
    g_in_weight: torch.Tensor,
    p_in_weight: torch.Tensor,
    _outgoing: bool,
    mask: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
) -> torch.Tensor:
    ##VS: similar issue here as to the cueq triangle attention
    ## kernel, we need to reshape the input so that batch and
    ## n_tmpl are combined into a single dimension.

    ## only hidden dimension multiple of 32 is supported for now
    if z.shape[-1] % 32 != 0:
        raise ValueError(
            "CUEQ triangle multiplicative update only supports "
            "channel dimension multiple of 32, got: "
            f"{z.shape[-1]}"
        )

    is_batched_input = False
    if len(z.shape) > 4:
        assert len(z.shape) == 5, (
            "CUEQ triangle multiplicative update only supports "
            f"max 5 input dimensions, got: {len(z.shape)}"
        )
        is_batched_input = True
        batch, n_tmpl, n_res, _, c_in = z.shape
        z = z.view(batch * n_tmpl, *z.shape[2:])
        mask = mask.view(batch * n_tmpl, *mask.shape[2:]) if mask is not None else None

    x = triangle_multiplicative_update(
        z,
        direction="outgoing" if _outgoing else "incoming",
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        g_in_weight=g_in_weight,
        p_in_weight=p_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=1e-5,
    )
    if is_batched_input:
        x = x.view(batch, n_tmpl, *x.shape[1:])
    return x

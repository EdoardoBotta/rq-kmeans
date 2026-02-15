import torch
import triton
import triton.language as tl
from torch.library import triton_op
from torch.library import wrap_triton

IS_PTX_RNA_TF32_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8

def kmeans_quantize(x: torch.Tensor, codebook: torch.Tensor):
    if x.shape[-1] <= 128 and x.shape[0] > 2*codebook.shape[0]:
        return quantize_fwd(x, codebook)
    return quantize_fwd_mm(x, codebook)[0]

@triton_op("kmeans::quantize", mutates_args={})
def quantize_fwd(
    x: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    assert x.shape[-1] == codebook.shape[-1]

    B, D = x.shape
    N, _ = codebook.shape

    quantized = torch.empty(B, dtype=torch.int32, device=x.device)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]),)
    wrap_triton(quantize_fwd_kernel)[grid](
        x,
        codebook,
        quantized,
        x.stride(0),
        x.stride(1),
        codebook.stride(0),
        codebook.stride(1),
        B,
        N,
        D,
        FP32_TO_TF32_MAX_PRECISION=IS_PTX_RNA_TF32_SUPPORTED,
    )

    return quantized

@triton.autotune(configs=[
    triton.Config({'BLOCK_B': 32,  'BLOCK_N': 32},  num_warps=4,  num_stages=2),
    triton.Config({'BLOCK_B': 64,  'BLOCK_N': 32},  num_warps=4,  num_stages=2),
    triton.Config({'BLOCK_B': 128, 'BLOCK_N': 32},  num_warps=8,  num_stages=2),
    triton.Config({'BLOCK_B': 64,  'BLOCK_N': 64},  num_warps=4,  num_stages=3),
    triton.Config({'BLOCK_B': 128, 'BLOCK_N': 64},  num_warps=8,  num_stages=3),
    triton.Config({'BLOCK_B': 256, 'BLOCK_N': 32},  num_warps=8,  num_stages=2),
  ],
  key=["B", "N", 'D'],
  restore_value=["out_ptr"]
)
@triton.jit
def quantize_fwd_kernel(
    x_ptr,
    codebook_ptr,
    out_ptr,
    x_stride_B,
    x_stride_D,
    codebook_stride_N,
    codebook_stride_D,
    B,
    N,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP32_TO_TF32_MAX_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offs_D = tl.arange(0, D)
    offs_N = tl.arange(0, BLOCK_N)
    offs_B = pid * BLOCK_B + tl.arange(0, BLOCK_B)

    x_ptrs = x_ptr + offs_B[:,None] * x_stride_B + offs_D[None,:] * x_stride_D
    mask = (offs_B[:,None] < B) & (offs_D[None,:] < D)
    x = tl.load(x_ptrs, mask=mask, other=0.0).cast(tl.float32)
    x_norm = tl.sum(x*x, axis=1)

    codebook_ptrs = codebook_ptr + offs_N[None, :] * codebook_stride_N + offs_D[:, None] * codebook_stride_D

    min_dist = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
    cluster = tl.full((BLOCK_B,), -1, dtype=tl.int32)
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        n_mask = (offs_N[None, :] < N - n * BLOCK_N)
        block_mask = n_mask & (offs_D[:, None] < D)
        
        codebook = tl.load(codebook_ptrs, mask=block_mask, other=0.0).cast(tl.float32)
        codebook_norm = tl.sum(codebook*codebook, axis=0)

        if FP32_TO_TF32_MAX_PRECISION:
            x = tl_fp32_to_tf32(x)
            codebook = tl_fp32_to_tf32(codebook)

        sim = tl.dot(x, codebook, allow_tf32=True)
        dist = tl.math.fma(sim, -2.0, x_norm[:, None] + codebook_norm[None, :])
        dist = tl.where(n_mask, dist, float("inf"))
        
        min_dist_batch, argmin_batch = tl.min(dist, axis=-1, return_indices=True)
        cluster_batch = n*BLOCK_N + argmin_batch

        cluster = tl.where(min_dist_batch < min_dist, cluster_batch, cluster)
        min_dist = tl.minimum(min_dist, min_dist_batch)

        codebook_ptrs += BLOCK_N * codebook_stride_N
    
    out_ptrs = out_ptr + offs_B
    tl.store(out_ptrs, cluster, mask=offs_B < B)

@triton.jit
def tl_fp32_to_tf32(x):
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1)

@triton_op("kmeans::quantize_mm", mutates_args={})
def quantize_fwd_mm(
    x: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == codebook.shape[-1]

    B, D = x.shape
    N, _ = codebook.shape

    quantized = torch.empty(B, dtype=torch.int32, device=x.device)
    dist = torch.full_like(quantized, float("inf"), dtype=torch.float32)
    locks = quantized.new_full((triton.cdiv(B, 16),), 0)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"])*triton.cdiv(N, meta["BLOCK_N"]),)
    wrap_triton(quantize_fwd_kernel_mm)[grid](
        x,
        codebook,
        quantized,
        dist,
        locks,
        locks.size(0),
        x.stride(0),
        x.stride(1),
        codebook.stride(0),
        codebook.stride(1),
        B,
        N,
        D,
        FP32_TO_TF32_MAX_PRECISION=IS_PTX_RNA_TF32_SUPPORTED,
    )

    return quantized, dist

@triton.autotune(configs=[
    triton.Config(
        {'BLOCK_B': 32, 'BLOCK_N': 32, 'BLOCK_D': 32, 'GROUP_SIZE_M': 8},
        num_warps=4, num_stages=3
    ),
    triton.Config(
        {'BLOCK_B': 64, 'BLOCK_N': 32, 'BLOCK_D': 32, 'GROUP_SIZE_M': 8},
        num_warps=8, num_stages=3
    ),
    triton.Config(
        {'BLOCK_B': 64, 'BLOCK_N': 64, 'BLOCK_D': 32, 'GROUP_SIZE_M': 8},
        num_warps=8, num_stages=3
    ),
    triton.Config(
        {'BLOCK_B': 64, 'BLOCK_N': 64, 'BLOCK_D': 64, 'GROUP_SIZE_M': 8},
        num_warps=8, num_stages=4
    ),
    triton.Config(
        {'BLOCK_B': 128, 'BLOCK_N': 64, 'BLOCK_D': 32, 'GROUP_SIZE_M': 4},
        num_warps=8, num_stages=3
    ),
    triton.Config(
        {'BLOCK_B': 128, 'BLOCK_N': 128, 'BLOCK_D': 32, 'GROUP_SIZE_M': 4},
        num_warps=8, num_stages=3
    ),
  ],
  key=["B", "N", 'D'],
  restore_value=["out_ptr", "dist_ptr", "locks_ptr"]
)
@triton.jit
def quantize_fwd_kernel_mm(
    x_ptr,
    codebook_ptr,
    out_ptr,
    dist_ptr,
    locks_ptr,
    num_locks,
    x_stride_B,
    x_stride_D,
    codebook_stride_N,
    codebook_stride_D,
    B,
    N,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP32_TO_TF32_MAX_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B, BLOCK_B)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_B = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_N = (pid % num_pid_in_group) // group_size_m
    
    offs_D = tl.arange(0, BLOCK_D)
    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_B[:,None] * x_stride_B + offs_D[None,:] * x_stride_D
    codebook_ptrs = codebook_ptr + offs_N[None, :] * codebook_stride_N + offs_D[:, None] * codebook_stride_D

    x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
    codebook_norm = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sim = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        x = tl.load(x_ptrs, mask=(offs_B[:,None] < B) & (offs_D[None,:] < D - d * BLOCK_D), other=0.0)
        codebook = tl.load(codebook_ptrs, mask=(offs_N[None,:] < N) & (offs_D[:,None] < D - d * BLOCK_D), other=0.0)
        
        x_norm += tl.sum(x*x, axis=1)
        codebook_norm += tl.sum(codebook*codebook, axis=0)

        if FP32_TO_TF32_MAX_PRECISION:
            x = tl_fp32_to_tf32(x)
            codebook = tl_fp32_to_tf32(codebook)
        sim = tl.dot(x, codebook, acc=sim, allow_tf32=True)

        x_ptrs += BLOCK_D * x_stride_D
        codebook_ptrs += BLOCK_D * codebook_stride_D
    
    dist_mask = (offs_B[:,None] < B) & (offs_N[None,:] < N)

    dist = tl.math.fma(sim, -2.0, x_norm[:, None] + codebook_norm[None, :])
    dist = tl.where(dist_mask, dist, float("inf"))

    block_min, argmin = tl.min(dist, axis=1, return_indices=True)
    
    locks_ptrs = locks_ptr + pid_B // tl.cdiv(B, BLOCK_B * num_locks)
    while tl.atomic_cas(locks_ptrs, 0, 1) == 1:
        pass

    prev = tl.atomic_min(dist_ptr + offs_B, block_min, mask=offs_B < B, sem="relaxed")
    improved = block_min < prev
    tl.store(out_ptr + offs_B, tl.where(improved, argmin + pid_N * BLOCK_N, tl.load(out_ptr + offs_B)), mask=offs_B < B)

    tl.debug_barrier()
    tl.atomic_xchg(locks_ptrs, 0)

quantize_fwd.register_kernel("cpu")
def quantize_cpu_fwd(x: torch.Tensor, codebook: torch.Tensor):
    _, quantized = torch.cdist(x, codebook).min(-1)
    return quantized

quantize_fwd_mm.register_kernel("cpu")
def quantize_cpu_fwd_mm(x: torch.Tensor, codebook: torch.Tensor):
    dist, quantized = torch.cdist(x, codebook).min(-1)
    return quantized, dist
    

    


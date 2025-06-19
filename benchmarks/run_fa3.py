from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

try:
    import cudnn
except ImportError:
    cudnn = None
# cudnn = None

Timing = NamedTuple('timing', [('mean', float)])


from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
# from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

import flash_attn_interface  # must be manually added to PYTHONPATH, see build_sm90.sh

from flash_attn_interface import flash_attn_func as flash_attn_func_v3
# from flash_attn_interface import flash_attn_with_kvcache as flash_attn_func_v3
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

from triton.testing import do_bench

try:
    from triton_fused_attention import attention as triton_attention
except ImportError:
    triton_attention = None
triton_attention = None

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


def time_fwd(func, *args, warmup=3, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=warmup, rep=repeats) * 1e-3)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q - window_size[1], torch.tensor(seqlen_k - 1))
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")




torch.manual_seed(0)

warmup = 0
repeats = 1

dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = 'cuda'
verbose = True
varlen = False
page_size = None
softcap = 0.0
V_colmajor = False
deterministic = False
batch_size = 2
# seqlen = 2048
seqlen = 8192
# seqlen = 4096
# seqlen = 2047
dim = 2048
# headdim = 128
# headdim = 64
headdim = 256
# for headdim in [64, 128, 256]:
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(32, 512), (16, 1024)]
# bs_seqlen_vals = [(2, 64 * 132)]
bs_seqlen_vals = [(2, 8192)]
# bs_seqlen_vals = [(1, 16 * 1024)]
time_f = {}
time_b = {}

# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [64, 96, 128]:
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192, 256]:
for headdim in [128]:
    nheads = dim // headdim
    # nheads = 128
    # headdim = 64
    # batch_size = 64
    # seqlen = 512
    # nheads = 8
    # headdim = 128
    nheads_kv = nheads
    # nheads_kv = nheads // 4
    # nheads_kv = 1
    headdim_v = headdim
    # headdim_v = 512
    has_qv = headdim == 64 and headdim_v == 512
    # has_qv = False

    for batch_size, seqlen in bs_seqlen_vals:
        num_splits = 0
        window_size = (-1, -1)
        # window_size = (seqlen // 2 - 1, 0)
        pack_gqa = None
        # seqlen_q = 64
        seqlen_q = seqlen
        leftpad_k = None
        # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in [q, k, v]]
        v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_()
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen) if has_qv else None
        # q = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # k = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # v = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim_v), device=device, dtype=torch.int32).to(dtype)
        g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        o = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        stats = torch.randn(batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32)
        if varlen:
            q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_() for x in [q, k, v]]
            cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
            cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
            # cu_seqlens_q = torch.tensor([0, 248, 249, 250, 251, 252, 253, 254, 255, 256], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:256]
            # seqlen_q = 256
            # cu_seqlens_q = torch.tensor([0, 376, 377, 378, 379, 380, 381, 382, 383, 384], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:384]
            # seqlen_q = 384
        if page_size is not None:
            assert seqlen % page_size == 0
            k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
            page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                   "(b s) -> b s", s=seqlen // page_size)
        else:
            page_table = None

        # for causal in [False, True]:
        for causal in [True]:
            print(f"\n### {headdim = }, {causal = }, {seqlen = } ###")
            if not varlen:
                out,_ = flash_attn_func_v3(q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, qv=qv, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa,)
                # m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, qv=qv, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, warmup=warmup, repeats=repeats, verbose=verbose, desc='Fav3')
            else:
                # m1 = time_fwd(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, warmup=warmup, repeats=repeats, verbose=verbose, desc='Fav3')
                out,_ = flash_attn_varlen_func_v3( q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa,)
            # time_f[(causal, headdim, batch_size, seqlen), "Flash3"] = m1.mean
            print('out: ', out.shape)
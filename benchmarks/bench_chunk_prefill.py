import random
import numpy as np
import gc
from eval.efficiency.generate_prompt import generate_prompt
import pickle
import torch
import time
from xattn.src.load_llama import load_fake_model, FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8, llama_fuse_16
from transformers import StaticCache
from tqdm import tqdm
import os
import math
import statistics
import argparse

import flash_attn_interface  # must be manually added to PYTHONPATH, see build_sm90.sh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vv", action="store_true", help="verbose")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-t", type=str, default=None)
    parser.add_argument("--th",
                        type=float,
                        default=0.9,
                        help="threshold for xattention.")

    # NOTE: change of model or dataset need to `rm -rf output`

    parser.add_argument("-d", type=str, default='default')

    # gradientai/Llama-3-8B-Instruct-Gradient-1048k
    # CohereLabs/aya-23-8B
    # CohereLabs/c4ai-command-r7b-12-2024 (need cohere's transformer, but still has problems)
    # mistralai/Mistral-7B-v0.1
    # mistralai/Ministral-8B-Instruct-2410 (ok after fix)
    #
    # deepseek-ai/DeepSeek-Prover-V2-7B (oom)
    #
    parser.add_argument("-m", type=str, default='CohereLabs/aya-23-8B')

    return parser.parse_args()


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles=None, return_mode='mean'):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func


def estimate_with_idx(
    chunk_idx,
    pad_query_states: torch.Tensor,
    pad_key_states: torch.Tensor,
    stride,
    k_block_num,
    q_block_num,
    #
    reshaped_block_size,
    reshaped_chunk_size,
    #
    k_reshaped_seq_len,
    k_reshaped_num_to_pad,
    head_dim,
    num_blocks_per_chunk,
    #
    norm=1,
    threshold=0.9,
    causal=True,
):
    attn_weights_slice = flat_group_gemm_fuse_reshape(
        pad_query_states[
            :,
            :,
            (chunk_idx * reshaped_chunk_size) * stride : \
                (chunk_idx * reshaped_chunk_size + reshaped_chunk_size) * stride,
            :,
        ],
        pad_key_states,
        stride,
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size,
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size
        + reshaped_chunk_size,
        is_causal=causal,
    )
    attn_sum = softmax_fuse_block_sum(
        attn_weights_slice,
        reshaped_block_size,
        min(4096, reshaped_block_size),
        (k_block_num - q_block_num) * reshaped_block_size +
        chunk_idx * reshaped_chunk_size,
        (k_block_num - q_block_num) * reshaped_block_size +
        chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
        k_reshaped_seq_len - k_reshaped_num_to_pad,
        1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
        is_causal=causal,
    )
    simple_mask = find_blocks_chunked(
        attn_sum,
        k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
        threshold,
        None,
        decoding=False,
        mode="prefill",
        causal=causal,
    )
    return attn_sum, simple_mask


def xattn_chunk_prefill(
    q,
    k,
    v,
    stride,
    block_size,
    threshold,
    use_triton,
    chunk_prefill_chunk_size,
    causal,
    ref_weight=None,
    ref_sums=None,
    ref_mask=None,
    ref_out=None,
):
    # 1. compute global meta data
    _, num_heads, k_len, _ = k.shape
    _, num_heads, q_len, _ = q.shape
    # global_q_block_num = (q_len + block_size - 1) // block_size
    # global_k_block_num = (k_len + block_size - 1) // block_size

    # 1.1 estimate meta data
    chunk_size = chunk_prefill_chunk_size
    assert q_len == k_len, f"q_len: {q_len}, k_len: {k_len}"
    assert q_len >= chunk_size and q_len % chunk_size == 0, f"q_len: {q_len}, chunk_size: {chunk_size}"
    num_chunks = q_len // chunk_size

    batch_size, num_kv_head, k_len, head_dim = k.shape
    batch_size, num_q_head, q_len, head_dim = q.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = (
        (k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = (
        (q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    if k_num_to_pad > 0:
        pad_key_states = F.pad(k, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = k
    if q_num_to_pad > 0:
        pad_query_states = F.pad(q, (0, 0, 0, q_num_to_pad),
                                 value=0).to("cuda")
    else:
        pad_query_states = q

    # 1.2 allocate memory
    global_q_view = torch.zeros_like(pad_query_states)
    global_k_view = torch.zeros_like(pad_key_states)
    global_v_view = torch.zeros_like(pad_key_states)

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []
    time_list = []

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    # 2. pretend that q, k, v is passed in as chunk

    ##2.1 chunk along seq dim (done by external)
    q_chunks = q.chunk(num_chunks, dim=2)
    k_chunks = k.chunk(num_chunks, dim=2)
    v_chunks = v.chunk(num_chunks, dim=2)

    # bench
    num_iterations = 100
    num_warmup = 20
    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    # iter over chunk
    for i, (q_chunk, k_chunk,
            v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        # insides the estimate_with_idx, it assumes a global view of the q, k
        # so we need to pad to global view, and put the q, k chunk in the right position
        global_q_view[:, :, i * chunk_size:(i + 1) * chunk_size, :] = q_chunk
        global_k_view[:, :, i * chunk_size:(i + 1) * chunk_size, :] = k_chunk
        global_v_view[:, :, i * chunk_size:(i + 1) * chunk_size, :] = v_chunk

        for _ in range(num_warmup):
            estimate_with_idx(
                i,
                global_q_view,
                global_k_view,
                stride,
                k_block_num,
                q_block_num,
                reshaped_block_size,
                reshaped_chunk_size,
                #
                k_reshaped_seq_len,
                k_reshaped_num_to_pad,
                head_dim,
                num_blocks_per_chunk,
                #
                threshold=threshold,
                causal=causal,
                norm=1,
            )
        torch.cuda.synchronize()
        start_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
        ]
        end_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
        ]

        for j in range(num_iterations):
            cache.zero_()
            start_event[j].record()
            sums, mask = estimate_with_idx(
                i,
                global_q_view,
                global_k_view,
                stride,
                k_block_num,
                q_block_num,
                reshaped_block_size,
                reshaped_chunk_size,
                #
                k_reshaped_seq_len,
                k_reshaped_num_to_pad,
                head_dim,
                num_blocks_per_chunk,
                #
                threshold=threshold,
                causal=causal,
                norm=1,
            )
            end_event[j].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        estimate_time = _summarize_statistics(times)

        attn_sum_list.append(sums)
        simple_mask_list.append(mask)
        time_list.append(estimate_time)

    # 3. after all chunks are processed
    # we gen full mask
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    assert causal
    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(q_block_num,
                           q_block_num,
                           dtype=bool,
                           device=pad_key_states.device),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )

    # 4. block-sparse
    assert block_size == 128
    assert batch_size == 1
    global_q_view = global_q_view.transpose(1, 2).view(q_len, num_heads,
                                                       head_dim)
    global_k_view = global_k_view.transpose(1, 2).view(k_len, num_heads,
                                                       head_dim)
    global_v_view = global_v_view.transpose(1, 2).view(k_len, num_heads,
                                                       head_dim)
    q_cu_seq_lens = torch.tensor([0, q_len],
                                 dtype=torch.int32,
                                 device=global_q_view.device)
    k_cu_seq_lens = torch.tensor([0, k_len],
                                 dtype=torch.int32,
                                 device=global_q_view.device)
    head_mask_type = torch.tensor([1 for _ in range(num_heads)],
                                  device=global_q_view.device,
                                  dtype=torch.int32)

    for _ in range(num_warmup):
        attn_output = block_sparse_attn_func(
            global_q_view,
            global_k_view,
            global_v_view,
            q_cu_seq_lens,
            k_cu_seq_lens,
            head_mask_type,
            None,
            simple_masks[:, :, :q_block_num, :k_block_num].contiguous(),
            q_len,
            k_len,
            p_dropout=0.0,
            deterministic=True,
            is_causal=causal,
        )
    torch.cuda.synchronize()
    start_event = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]

    for j in range(num_iterations):
        cache.zero_()
        start_event[j].record()
        attn_output = block_sparse_attn_func(
            global_q_view,
            global_k_view,
            global_v_view,
            q_cu_seq_lens,
            k_cu_seq_lens,
            head_mask_type,
            None,
            simple_masks[:, :, :q_block_num, :k_block_num].contiguous(),
            q_len,
            k_len,
            p_dropout=0.0,
            deterministic=True,
            is_causal=causal,
        )
        end_event[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    block_sparse_time = _summarize_statistics(times)
    attn_output = attn_output.view(batch_size, q_len, num_heads,
                                   head_dim).transpose(1, 2)

    # check if ref provided
    if ref_weight is not None:
        atol = 1e-2
        rtol = 0
        if torch.allclose(attn_sums, ref_sums, atol=atol, rtol=rtol):
            print("✅ sums match")
        else:
            print("❌ sums differ")
        if torch.allclose(simple_masks, ref_mask, atol=atol, rtol=rtol):
            print("✅ mask match")
        else:
            print("❌ mask differ")
        if torch.allclose(attn_output, ref_out, atol=atol, rtol=rtol):
            print("✅ attn out match")
        else:
            print("❌ attn out differ")
    print(f'dynamic mask times: ')
    for t in time_list:
        print(f'{t:.2f}ms', end=', ')
    print()
    print(f'block sparse times: {block_sparse_time:.2f}ms')
    return attn_output


# #################################
# #################################
# #################################
# #################################


def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    #
    # the attn map is q_len x q_len
    # then it is divided into [q_len // block_size, q_len // block_size] <- num of blocks
    # then each block can be divided into [q_len // block_size // stride, q_len // block_size // stride] <- num of sub-blocks
    #

    k_num_to_pad = (
        (k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = (
        (q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad),
                               value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad),
                                 value=0).to("cuda")
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    if use_triton and ("100" not in torch.cuda.get_device_properties(
            torch.cuda.current_device()).name):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :])
                                      for k in range(stride)],
                                     dim=-1)
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i]::stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :])
                                      for k in range(stride)],
                                     dim=-1)
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, (stride - 1 - q)::(stride * kdb), :])
                 for q in range(stride)],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :])
                                      for k in range(stride)],
                                     dim=-1)
            reshaped_query = torch.cat([(pad_query_states[:, :, q::stride, :])
                                        for q in range(stride)],
                                       dim=-1)
        elif select_mode == "double":
            reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :])
                                      for k in range(stride)],
                                     dim=-1)
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :,
                                                                   0:head_dim]
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, (stride - 1 - q)::stride, :])
                 for q in range(stride)],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :])
                                      for k in range(stride)],
                                     dim=-1)
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :,
                                                                   0:head_dim]
                ],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, -head_dim:],
                    reshaped_key[:, :, :, 0:-head_dim]
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, (stride - 1 - q)::stride, :])
                 for q in range(stride)],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size) *
                    stride:(chunk_idx * reshaped_chunk_size +
                            reshaped_chunk_size) * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size +
                chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size +
                chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size +
                chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size +
                chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) //
                kdb:(chunk_idx * reshaped_chunk_size + reshaped_chunk_size) //
                kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to("cuda")

            attn_weights_slice = (attn_weights_slice / math.sqrt(head_dim) /
                                  stride / norm)

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = chunk_idx * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    ) * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (
                        -(q_reshaped_num_to_pad // kdb)):, :] = float("-inf")

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1::kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device)

            if softmax:
                attn_weights_slice = F.softmax(attn_weights_slice,
                                               dim=-1,
                                               dtype=torch.float32).to(
                                                   pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype)
            attn_weights_slice = F.dropout(attn_weights_slice,
                                           p=0,
                                           training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :,
                                   (-(q_reshaped_num_to_pad // kdb)):, :] = 0

            attn_sum = (attn_weights_slice.view(
                batch_size,
                num_kv_head,
                num_blocks_per_chunk,
                reshaped_block_size // kdb,
                -1,
                reshaped_block_size,
            ).sum(dim=-1).sum(dim=-2).to("cuda"))
            del chunked_query

        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        # del attn_weights_slice # XXX why del?

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(q_block_num,
                           q_block_num,
                           dtype=bool,
                           device=key_states.device),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )

    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num,
                               device=simple_masks.device,
                               dtype=bool)
        eye_matrix_expanded = (eye_matrix.unsqueeze(0).unsqueeze(0).expand(
            1, num_kv_head, q_block_num, q_block_num))
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:,
                                                    -q_block_num:])

    return attn_weights_slice, attn_sums, simple_masks


def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            ))

    attn_weights_slice, attn_sums, approx_simple_mask = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    # for hid in range(num_heads):
    #     print(approx_simple_mask[0, hid, :, :].sum(), end=' ')
    # print()

    ####################
    assert block_size == 128
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_heads,
                                                     head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_heads,
                                                     head_dim)
    q_cu_seq_lens = torch.tensor([0, q_len],
                                 dtype=torch.int32,
                                 device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len],
                                 dtype=torch.int32,
                                 device=query_states.device)
    head_mask_type = torch.tensor([1 for _ in range(num_heads)],
                                  device=query_states.device,
                                  dtype=torch.int32)
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )

    ################################
    attn_output = attn_output.view(batch_size, q_len, num_heads,
                                   head_dim).transpose(1, 2)
    # del query_states
    # num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    # del approx_simple_mask,
    return attn_output, attn_weights_slice, attn_sums, approx_simple_mask


def bench_fa(q, k, v, num_warmups, num_iterations, cache):
    for i in range(num_warmups):
        # bs, nhead, seqlen, headim -> (batch_size, seqlen, nheads, headdim)
        flash_attn_interface.flash_attn_func(q.permute(0, 2, 1, 3),
                                             k.permute(0, 2, 1, 3),
                                             v.permute(0, 2, 1, 3),
                                             softmax_scale=None,
                                             causal=True)

    # For flash attention
    # permute outside of timer
    q_flash, k_flash, v_flash = q.permute(0, 2, 1,
                                          3), k.permute(0, 2, 1, 3), v.permute(
                                              0, 2, 1, 3)
    torch.cuda.synchronize()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(num_iterations)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True) for i in range(num_iterations)
    ]
    for i in range(num_iterations):
        cache.zero_()
        start_event[i].record()
        o, softmax_lse = flash_attn_interface.flash_attn_func(
            q_flash, k_flash, v_flash, softmax_scale=None, causal=True)
        end_event[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    avg_time_flash_attn = _summarize_statistics(times)

    del o
    del q_flash, k_flash, v_flash
    gc.collect()
    return avg_time_flash_attn


def bench_xa(
    q,
    k,
    v,
    num_warmups,
    num_iterations,
    cache,
    # xattn args
    stride,
    threshold,
    chunk_size,
):
    for i in range(num_warmups):
        Xattention_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,

            # unify chunk_size
            chunk_size=chunk_size,
        )

    # For flash attention
    # permute outside of timer
    torch.cuda.synchronize()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(num_iterations)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True) for i in range(num_iterations)
    ]
    for i in range(num_iterations):
        cache.zero_()
        start_event[i].record()
        ref_out, ref_weight, ref_sums, ref_mask = Xattention_prefill(
            q,
            k,
            v,
            stride=stride,
            threshold=threshold,
            use_triton=True,

            # unify chunk_size
            chunk_size=chunk_size,
        )
        end_event[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    avg_time = _summarize_statistics(times)
    gc.collect()
    return avg_time


def main():
    args = parse_args()
    # lens = [8, 16, 32, 64]
    lens = [512, 1024]
    if args.full:
        lens = [8, 16, 32, 64, 128, 256, 512, 768,] #1024]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Model: {args.m}, Dataset: {args.d}, context length: {lens}')
    device = torch.device("cuda:0")
    chunk_size = 4096

    for length in lens:
        #
        # GEN
        #
        print(f"Testing {length}K")
        query_path = f"output/query_{length*1024}.pkl"
        key_path = f"output/key_{length*1024}.pkl"
        layer_to_save = 12

        if not os.path.exists(query_path) or not os.path.exists(key_path):
            print(f'[NEW Q, K, V]')
            past_key_values = None # we always allocate new one for new length
            torch.cuda.empty_cache()

            # model, tokenizer = load_fake_model(name_or_path="meta-llama/Llama-3.1-8B-Instruct", layer_to_save=layer_to_save, target_len=len*1024)
            model, tokenizer = load_fake_model(
                name_or_path=args.m,
                layer_to_save=layer_to_save,
                target_len=length * 1024,
                token=args.t,
                cut=True, # only use first layer
            )
            input_ids = generate_prompt(tokenizer,
                                        length * 1024,
                                        datasets=args.d,
                                    )
            model = model.to(device)
            input_ids = input_ids.to(device)

            if past_key_values is not None:
                past_key_values.reset()
            else:
                # XXX: normal transformers
                past_key_values = StaticCache(
                    config=model.config,
                    batch_size=1,
                    max_cache_len=length * 1024,
                    device=model.device,
                    dtype=model.dtype,
                )

            # XXX: cohere transformers
            # past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=300000, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                # for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling", unit="chunk"):
                for i in range(0, input_ids.size(1), chunk_size):
                    chunk = input_ids[:, i:i + chunk_size]
                    # print(chunk.shape)
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
        # discard kv cache
        past_key_values = None 
        torch.cuda.empty_cache()

        #
        # BENCH
        #
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)
        assert (
            q.shape[-2] == length *
            1024), f"q.shape[-2]: {q.shape[-2]}, length*1024: {length*1024}"
        assert (
            k.shape[-2] == length *
            1024), f"k.shape[-2]: {k.shape[-2]}, length*1024: {length*1024}"

        q = q.to(device)
        k = k.to(device)
        v = torch.randn(q.shape, dtype=torch.bfloat16).to(device).contiguous()

        num_iterations = 100
        num_warmup = 20
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

        q_len = q.shape[-2]
        num_chunks = q_len // chunk_size

        print(f"len :{length}K\n"
              f"q.shape: {q.shape}, {q.dtype}\n"
              f"k.shape: {k.shape}, {k.dtype}\n"
              f"v.shape: {v.shape}, {v.dtype}\n"
              f'num_chunks: {num_chunks}\n')
        #
        # FA
        #
        q_chunks = q.chunk(num_chunks, dim=2)
        k_chunks = k.chunk(num_chunks, dim=2)
        v_chunks = v.chunk(num_chunks, dim=2)
        kv_cache = []

        # iter over chunk
        fa_times = []
        for i, (q_chunk, k_chunk,
                v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
            # 1. Append current k_chunk and v_chunk to the cache
            kv_cache.append((k_chunk, v_chunk))

            # 2. Prepare k_all and v_all by concatenating all items in kv_cache
            # Extract all k's and v's from the cache
            k_list_from_cache = [item[0] for item in kv_cache]
            v_list_from_cache = [item[1] for item in kv_cache]
            k_all = torch.cat(k_list_from_cache, dim=2)
            v_all = torch.cat(v_list_from_cache, dim=2)

            # TODO q_len != k_len just works? fa natively supports q_len != k_len?
            fa_time = bench_fa(q_chunk, k_all, v_all, num_iterations,
                               num_warmup, cache)
            fa_times.append(fa_time)
        if args.vv:
            for i, fa_time in enumerate(fa_times):
                print(f"FA chunk {i}: {fa_time:.2f}ms")

        #
        # Sparse-attn
        #
        # threshold = torch.tensor(llama_fuse_8)[layer_to_save]
        threshold = args.th  # NOTE: TUNE for model accuracy and speed

        x16_times = []
        x8_times = []

        # for stride in [8, 16]:
        for stride in [16, 8]:
            print(
                f'Stride: {stride}, chunk_size: {chunk_size}, threshold: {threshold}'
            )
            q_chunks = q.chunk(num_chunks, dim=2)
            k_chunks = k.chunk(num_chunks, dim=2)
            v_chunks = v.chunk(num_chunks, dim=2)
            kv_cache = []

            # iter over chunk
            for i, (q_chunk, k_chunk,
                    v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
                # 1. Append current k_chunk and v_chunk to the cache
                kv_cache.append((k_chunk, v_chunk))

                # 2. Prepare k_all and v_all by concatenating all items in kv_cache
                # Extract all k's and v's from the cache
                k_list_from_cache = [item[0] for item in kv_cache]
                v_list_from_cache = [item[1] for item in kv_cache]
                k_all = torch.cat(k_list_from_cache, dim=2)
                v_all = torch.cat(v_list_from_cache, dim=2)

                # TODO when q_len!=k_len, the padding Q direction is wrong?? 
                xa_time = bench_xa(
                    q_chunk,
                    k_all,
                    v_all,
                    num_iterations,
                    num_warmup,
                    cache,
                    stride=stride,
                    threshold=threshold,
                    chunk_size=chunk_size,
                )
                if stride == 16:
                    x16_times.append(xa_time)
                elif stride == 8:
                    x8_times.append(xa_time)
            if args.vv:
                for i, x16_time in enumerate(x16_times):
                    print(f"X16 chunk {i}: {x16_time:.2f}ms")
                for i, x8_time in enumerate(x8_times):
                    print(f"X8 chunk {i}: {x8_time:.2f}ms")

            #
            # VERIFY TODO one-shot should match sequential, but possible?
            #

            # ref_out, ref_weight, ref_sums, ref_mask = Xattention_prefill(q, k, v,
            #                                                         stride=stride,
            #                                                         threshold=threshold,
            #                                                         use_triton=True,

            #                                                         # unify chunk_size
            #                                                         chunk_size=chunk_size,
            #                                                         )

            # q_chunks = q.chunk(num_chunks, dim=2)
            # k_chunks = k.chunk(num_chunks, dim=2)
            # v_chunks = v.chunk(num_chunks, dim=2)
            # kv_cache = []

            # # iter over chunk
            # for i, (q_chunk, k_chunk, v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
            #     # 1. Append current k_chunk and v_chunk to the cache
            #     kv_cache.append((k_chunk, v_chunk))

            #     # 2. Prepare k_all and v_all by concatenating all items in kv_cache
            #     # Extract all k's and v's from the cache
            #     k_list_from_cache = [item[0] for item in kv_cache]
            #     v_list_from_cache = [item[1] for item in kv_cache]
            #     k_all = torch.cat(k_list_from_cache, dim=2)
            #     v_all = torch.cat(v_list_from_cache, dim=2)

            #     chunk_out = xattn_chunk_prefill(q_chunk, k_all, v_all,
            #                                 stride,
            #                                 block_size=128,
            #                                 threshold=threshold,
            #                                 use_triton=True,
            #                                 chunk_prefill_chunk_size=chunk_size,
            #                                 causal=True,
            # )

        fa = sum(fa_times) / len(fa_times)
        x16 = sum(x16_times) / len(x16_times)
        x8 = sum(x8_times) / len(x8_times)
        print(f"FA: {fa:.2f}ms, X16: {x16:.2f}ms, X8: {x8:.2f}ms")
        print('*' * 120)
        # break


if __name__ == "__main__":
    main()

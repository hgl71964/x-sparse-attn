
"""
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False):
dropout_p should be set to 0.0 during evaluation
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
Return:
    out: (batch_size, seqlen, nheads, headdim).
"""
import gc
from xattn.src.Xattention import Xattention_prefill
XATTN_PREFILL = True
# try:
#     from xattn.src.Xattention import Xattention_prefill
#     XATTN_PREFILL = True
# except:
#     XATTN_PREFILL = False

try:
    from xattn.src.Flexprefill import Flexprefill_prefill
    FLEXPREFILL_PREFILL = True
except:
    FLEXPREFILL_PREFILL = False

try:
    from xattn.src.Minference import Minference_prefill
    MINFERENCE_PREFILL = True
except:
    MINFERENCE_PREFILL = False

try:
    from xattn.src.Fullprefill import Full_prefill
    FULL_PREFILL = True
except:
    FULL_PREFILL = False
# from flash_attn import flash_attn_interface  # for flash attn 2
# from flash_attn_3 import flash_attn_interface # this import seems to be broken: https://github.com/Dao-AILab/flash-attention/issues/1536
import flash_attn_interface  # must be manually added to PYTHONPATH, see build_sm90.sh
import pickle
import torch
import time
from eval.efficiency.generate_prompt import generate_prompt
from xattn.src.load_llama import load_fake_model,FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8,llama_fuse_16
from transformers import StaticCache
from tqdm import tqdm
import os

if __name__ == "__main__":

    lens = [4,8,16,32,64,128]
    # lens = [4,8]

    speedups_flex = []
    speedups_xattn_8 = []
    speedups_xattn_16 = []
    speedups_minfer = []
    past_key_values = None
    for len in lens:
        print(f"Testing {len}K")
        query_path = f"output/query_{len*1024}.pkl"
        key_path = f"output/key_{len*1024}.pkl"
        config = FastPrefillConfig(metric = "xattn",stride = 16)
        layer_to_save = 12
        if not os.path.exists(query_path) or not os.path.exists(key_path):

            #model, tokenizer = load_fake_model(name_or_path="meta-llama/Llama-3.1-8B-Instruct", layer_to_save=layer_to_save, target_len=len*1024)
            model, tokenizer = load_fake_model(name_or_path="gradientai/Llama-3-8B-Instruct-Gradient-1048k", layer_to_save=layer_to_save, target_len=len*1024)
            input_ids = generate_prompt(tokenizer,len*1024)
            chunk_size = 4096
            if past_key_values is not None:
                past_key_values.reset()
            else:
                past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=300000, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling", unit="chunk"):
                    chunk = input_ids[:, i: i + chunk_size]
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)
        assert(q.shape[-2] == len*1024)
        assert(k.shape[-2] == len*1024)
        torch.manual_seed(0)
        # FlexPrefill args
        gamma = 0.95
        tau = 0.1
        # Xattention args
        threshold = torch.tensor(llama_fuse_8)[layer_to_save]
        stride = 16
        v = torch.randn(q.shape, dtype=torch.bfloat16).to("cuda").contiguous()
        num_iterations = 50
        num_warmups = 30
        # warm up
        print(f"len :{len}\n"
              f"q.shape: {q.shape}\n"
              f"k.shape: {k.shape}\n"
              f"v.shape: {v.shape}\n"
        )
        for i in range(num_warmups):
            try:
                Xattention_prefill(q, k, v, stride=16, threshold=threshold, use_triton=True)
                Xattention_prefill(q, k, v, stride=8, threshold=threshold, use_triton=True)
            except:
                XATTN_PREFILL = False
            try:
                Full_prefill(q, k, v, causal=False)
            except:
                FULL_PREFILL = False
            try:
                Flexprefill_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gamma, tau)
            except:
                FLEXPREFILL_PREFILL = False
            try:
                Minference_prefill(k, q, v)
            except:
                MINFERENCE_PREFILL = False
            # bs, nhead, seqlen, headim -> (batch_size, seqlen, nheads, headdim)
            _ = flash_attn_interface.flash_attn_func(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3), softmax_scale=None, causal=True)

        # Efficiency Evaluation
        # For Flexprefill_prefill
        total_time_flex = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if FLEXPREFILL_PREFILL:
                flex_prefill_output = Flexprefill_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gamma, tau)
            else:
                FLEXPREFILL_PREFILL = False
            torch.cuda.synchronize()
            total_time_flex += time.time() - start_time
        avg_time_flex = total_time_flex / num_iterations
        # del flex_prefill_output
        # gc.collect()

        # For Xattention_prefill
        total_time_xattn_8 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            flex_prefill_output = Xattention_prefill(q, k, v, stride=8, threshold= threshold, use_triton=True,chunk_size=min(32768,len*1024))
            torch.cuda.synchronize()
            total_time_xattn_8 += time.time() - start_time
        avg_time_xattn_8 = total_time_xattn_8 / num_iterations
        del flex_prefill_output
        gc.collect()

        total_time_xattn_16 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            flex_prefill_output = Xattention_prefill(q, k, v, stride=16, threshold= threshold, use_triton=True,chunk_size=min(32768,len*1024))
            torch.cuda.synchronize()
            total_time_xattn_16 += time.time() - start_time
        avg_time_xattn_16 = total_time_xattn_16 / num_iterations
        del flex_prefill_output
        gc.collect()

        # For minference
        total_time_minfer = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if MINFERENCE_PREFILL:
                try:
                    flex_prefill_output = Minference_prefill(k, q, v)
                except:
                    MINFERENCE_PREFILL = False
            torch.cuda.synchronize()
            total_time_minfer += time.time() - start_time
        avg_time_minfer = total_time_minfer / num_iterations
        # del flex_prefill_output
        # gc.collect()

        # For flash attention
        total_time_flashinfer = 0
        torch.cuda.synchronize()
        # permute outside of timer
        q_flash, k_flash, v_flash = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
        start_time = time.time()
        # o = Full_prefill(q, k, v, causal=False)
        try:
            o = flash_attn_interface.flash_attn_func(q_flash, k_flash, v_flash, softmax_scale=None, causal=True)
        except:
            continue
        torch.cuda.synchronize()
        total_time_flashinfer += time.time() - start_time
        avg_time_fa = total_time_flashinfer
        del o
        del q_flash, k_flash, v_flash
        gc.collect()

        # Calculate speedups -> # full here is fa3
        print(f"{len}K Minfer {avg_time_minfer:.4f} flex: {avg_time_flex:.4f} xattn_8: {avg_time_xattn_8:.4f} xattn_16: {avg_time_xattn_16:.4f} full(fa): {avg_time_fa:.4f} ")
        speedup_flex = avg_time_fa / avg_time_flex
        speedup_xattn_8 = avg_time_fa / avg_time_xattn_8
        speedup_xattn_16 = avg_time_fa / avg_time_xattn_16
        speedup_minfer = avg_time_fa / avg_time_minfer
        speedups_flex.append(speedup_flex)
        speedups_xattn_8.append(speedup_xattn_8)
        speedups_xattn_16.append(speedup_xattn_16)
        speedups_minfer.append(speedup_minfer)

    # Output results
    #print(f"\n{'Length':<10}{'Flex Speedup':<15}{'Xattn 8 Speedup':<20}{'Xattn 16 Speedup':<25}{'Minfer Speedup'}")
    #for len, speedup_flex, speedup_xattn_8, speedup_xattn_16,speedup_minfer in zip(lens, speedups_flex, speedups_xattn_8, speedups_xattn_16, speedups_minfer):
    #    print(f"{str(len):<10}{speedup_flex:<15.2f}{speedup_xattn_8:<20.2f}{speedup_xattn_16:<25.2f}{speedup_minfer:.2f}")
    #print(f"\nMike:{'Length':<10}{'Xattn 8 Speedup':<30}{'Xattn 16 Speedup':<35}")
    #for len, speedup_flex, speedup_xattn_8, speedup_xattn_16,speedup_minfer in zip(lens, speedups_flex, speedups_xattn_8, speedups_xattn_16, speedups_minfer):
    #    l = str(len) + 'K'
    #    print(f"{l:<10}{speedup_xattn_8:<30.2f}{speedup_xattn_16:<35.2f}")

    # Header
    print(f"\n{'Mike:':<10}{'Length':<10}{'Xattn 8 Speedup':<20}{'Xattn 16 Speedup':<20}")

    # Data rows
    for length, speedup_flex, speedup_xattn_8, speedup_xattn_16, speedup_minfer in zip(
        lens, speedups_flex, speedups_xattn_8, speedups_xattn_16, speedups_minfer
    ):
        length_str = f"{length}K"
        print(f"{'':<10}{length_str:<10}{speedup_xattn_8:<20.2f}{speedup_xattn_16:<20.2f}")

import random

import torch
from datasets import load_dataset

def generate_prompt(tokenizer,target_len,datasets='default'):
    if datasets == 'default':
        input_ids = _default(tokenizer, target_len)
    elif datasets == 'longbench':
        input_ids = _long_bench(tokenizer, target_len, task_filter=None)
    else:
        raise RuntimeError(f"Unknown dataset: {datasets}")

    return input_ids

def _default(tokenizer, target_len: int):
    context = "A quick brown fox jumps over the lazy dog. \n"
    with open("demo/xattention.txt", "r") as f:
        needle = f.read()

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = target_len // num_tokens_context

    text = (
        "This is a very long story book with knowledge of XAttention, which you need to remember for later question: <book> "
        + context * int(num_repetitions * 0.5)
        + needle
        + context * int(num_repetitions * 0.5)
        + "</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:"
    )

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    suffix_len = len(tokenizer("</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:", add_special_tokens=False))
    over_len = input_ids.shape[1] - target_len
    input_ids = torch.cat([input_ids[:, :-suffix_len-100-over_len], input_ids[:, -suffix_len-100:]], dim=1) if over_len > 0 else input_ids
    return input_ids

#########################################
######################################### long bench
#########################################
def _long_bench(tokenizer, target_len, task_filter=None):
    dataset_version = "v1"
    sub_dataset="narrativeqa samsum qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
    sub_dataset = sub_dataset.split()
    sub_dataset = sub_dataset[0]

    try:
        if dataset_version == "v2":
            # data = load_dataset('THUDM/LongBench-v2', split='train')
            raise 
        elif dataset_version == "v1":
            data = load_dataset('THUDM/LongBench', sub_dataset, split='test')
        else:
            raise ValueError(f"Unsupported dataset version: {dataset_version}")
            
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")
    
    tolerance = 0.05
    lower_bound = int(target_len * (1 - tolerance))
    upper_bound = int(target_len * (1 + tolerance))
    
    for i, instance in enumerate(data):
        prompt = instance.get('input')

        # Tokenize the prompt
        # - padding='max_length': pads the sequence to target_len if it's shorter.
        # - truncation=True: truncates the sequence to target_len if it's longer.
        # - max_length=target_len: specifies the target length.
        # - return_tensors='pt': returns PyTorch tensors.
        tokenized_output = tokenizer(
            prompt,
            # max_length=target_len,
            # padding="max_length",
            # truncation=True,
            return_tensors="pt",
            add_special_tokens=True, # Usually True, adds [CLS], [SEP] etc.
        )

        # input_ids will be a 2D tensor of shape [1, target_len] because we processed a single string.
        # We can squeeze it to get a 1D tensor.
        input_ids = tokenized_output['input_ids'].squeeze(0)

        if lower_bound <= len(input_ids) <= upper_bound:
            print(f'{lower_bound=}, {len(input_ids)=}, {upper_bound=}')
            return input_ids

    raise ValueError(f"Failed to generate prompt with target length {target_len}")
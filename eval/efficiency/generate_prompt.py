import random

import torch
from datasets import load_dataset

def generate_prompt(tokenizer,target_len,datasets='default'):
    if datasets == 'default':
        input_ids = _default(tokenizer, target_len)
    elif datasets == 'longbench':
        input_ids = _long_bench_v1(tokenizer, target_len, task_filter=None)
    else:
        raise RuntimeError(f"Unknown dataset: {datasets}")

    return input_ids

#########################################
######################################### default
#########################################
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
def _long_bench_v1(tokenizer, target_len, task_filter=None):

    def build_chat(inputs, context):
        prompt = f"Context: {context}\n\nQuestion: {inputs}\n\nAnswer:"
        return prompt

    sub_datasets= ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for sub_dataset in sub_datasets:
        data = load_dataset('THUDM/LongBench', sub_dataset, split='test')
                
        tolerance = 0.05
        lower_bound = int(target_len * (1 - tolerance))
        upper_bound = int(target_len * (1 + tolerance))
        
        for i, instance in enumerate(data):
            inputs = instance.get('input')
            context = instance.get('context')
            prompt = build_chat(inputs, context)

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
            # input_ids = tokenized_output['input_ids'].squeeze(0)
            input_ids = tokenized_output['input_ids']

            # print(f'{sub_dataset}, {i=}, {input_ids.shape}')
            if lower_bound <= input_ids.shape[1] <= upper_bound:
                # print(f'{lower_bound=}, {input_ids.shape}, {upper_bound=}')

                # make it exact
                tokenized_output = tokenizer(
                    prompt,
                    max_length=target_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True, 
                )
                input_ids = tokenized_output['input_ids'].to("cuda")
                return input_ids

    raise ValueError(f"Failed to generate prompt with target length {target_len}")
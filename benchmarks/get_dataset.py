from transformers import AutoTokenizer # For example
from datasets import load_dataset     # For example usage
import torch                          # Assuming you want PyTorch tensors

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default='gradientai/Llama-3-8B-Instruct-Gradient-1048k')
    return parser.parse_args()

def generate_input_ids(prompt_text: str, tokenizer, target_len: int):
    """
    Tokenizes a given prompt string to input_ids,
    applying padding and truncation to meet the target_len.

    Args:
        prompt_text (str): The text prompt to tokenize.
        tokenizer: An instance of a Hugging Face tokenizer
                   (e.g., from AutoTokenizer.from_pretrained()).
        target_len (int): The desired length for the tokenized output.
                          Sequences shorter will be padded, longer will be truncated.

    Returns:
        torch.Tensor: A 1D PyTorch tensor of input_ids.
                      Returns None if prompt_text is empty or None.
    """
    if not prompt_text:
        print("Warning: Received empty or None prompt_text.")
        return None

    # Tokenize the prompt
    # - padding='max_length': pads the sequence to target_len if it's shorter.
    # - truncation=True: truncates the sequence to target_len if it's longer.
    # - max_length=target_len: specifies the target length.
    # - return_tensors='pt': returns PyTorch tensors.
    tokenized_output = tokenizer(
        prompt_text,
        max_length=target_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True # Usually True, adds [CLS], [SEP] etc.
    )

    # input_ids will be a 2D tensor of shape [1, target_len] because we processed a single string.
    # We can squeeze it to get a 1D tensor.
    # input_ids = tokenized_output['input_ids'].squeeze(0)
    input_ids = tokenized_output['input_ids']

    return input_ids

def examine(tokenizer):
    # 1. Choose a sub-dataset from LongBench
    # (Make sure you have `datasets` installed: pip install datasets)
    # (And `transformers` and `torch`: pip install transformers torch)
    # longbench_sub_dataset = "narrativeqa" # Example, pick one you want to test
    longbench_sub_dataset = "lcc" # Example, pick one you want to test

    print(f"Loading dataset: THUDM/LongBench, config: {longbench_sub_dataset}")
    try:
        data = load_dataset("THUDM/LongBench", longbench_sub_dataset, split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the sub-dataset name is correct and you have internet access.")
        exit()

    # 3. Define a target length for tokenized sequences
    example_target_len = 128 # A small target length for demonstration

    # 4. Iterate through a few instances and generate input_ids
    print(f"\nGenerating input_ids for the first few instances (target_len={example_target_len}):")
    for i, instance in enumerate(data):
        if i >= 3: # Process only the first 3 instances for this example
            break

        # Extract the prompt (usually in the 'input' field for LongBench)
        prompt_from_instance = instance.get('input')

        if prompt_from_instance:
            print(f"\n--- Instance {i+1} ---")
            print(f"Original Prompt (first 100 chars): {prompt_from_instance[:100]}...")

            # Generate input_ids using the function
            input_ids_tensor = generate_input_ids(prompt_from_instance, tokenizer, example_target_len)

            if input_ids_tensor is not None:
                print(f"Generated input_ids (shape: {input_ids_tensor.shape}):")
                # print(input_ids_tensor) # Uncomment to see the full tensor
                print(f"First 10 IDs: {input_ids_tensor[:10].tolist()}...")
                print(f"Length of input_ids: {len(input_ids_tensor)}")

                # Optional: Decode back to see how truncation/padding affected it
                decoded_text = tokenizer.decode(input_ids_tensor, skip_special_tokens=False)
                print(f"Decoded back (first 100 chars, with special tokens): {decoded_text[:100]}...")
                decoded_text_no_special = tokenizer.decode(input_ids_tensor, skip_special_tokens=True)
                print(f"Decoded back (first 100 chars, no special tokens): {decoded_text_no_special[:100]}...")
        else:
            print(f"\n--- Instance {i+1} ---")
            print("Could not find 'input' field in this instance.")
            print(f"Available keys: {instance.keys()}")

    print("\nExample finished.")

def build_chat(inputs, context):
    prompt = f"Context: {context}\n\nQuestion: {inputs}\n\nAnswer:"
    return prompt


def print_all(tokenizer):
    sub_datasets= ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    for sub_dataset in sub_datasets:
        data = load_dataset('THUDM/LongBench', sub_dataset, split='test')
        print(f'Loading dataset: THUDM/LongBench, config: {sub_dataset}, len: {len(data)}')

        for i, instance in enumerate(data):
            inputs = instance.get('input')
            context = instance.get('context')
            prompt = build_chat(inputs, context)

            tokenized_output = tokenizer(
                prompt,
                # max_length=target_len,
                # padding="max_length",
                # truncation=True,
                return_tensors="pt",
                add_special_tokens=True, # Usually True, adds [CLS], [SEP] etc.
            )
            input_ids = tokenized_output['input_ids']
            print(f'{input_ids.shape}, ',end='')
        print()

def ruler(tokenizer):
    # ruler_data_for_tokenization = load_dataset("NVIDIA/RULER", sub_dataset, split="test")
    data = load_dataset("NVIDIA/RULER", split="test")
    print(data.keys())


def main():
    args = parse_args()
    print(f'model: {args.m}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.m
    )
    examine(tokenizer)
    # print_all(tokenizer)
    # ruler(tokenizer)

if __name__ == '__main__':
    main()
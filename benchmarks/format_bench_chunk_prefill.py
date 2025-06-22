import re
from typing import Dict, List, Any
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, default=None)
    return parser.parse_args()

def parse_attention_log(log_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse attention performance log and extract metrics into a structured dictionary.

    Args:
        log_text: The raw log text as a string

    Returns:
        Dictionary with sequence lengths as keys and performance data as values
    """
    results = {}

    # Split log into sections by asterisks separator
    sections = log_text.split('*' * 120)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')

        # Extract sequence length from the first line
        seq_length = None
        for line in lines:
            if line.startswith('Testing') and 'K' in line:
                # Extract sequence length (e.g., "16K", "32K", "64K")
                match = re.search(r'Testing (\d+K)', line)
                if match:
                    seq_length = match.group(1)
                    break

        if not seq_length:
            continue

        # Extract average latencies
        avg_latencies = {}
        chunk_data = {'FA': [], 'X16': [], 'X8': []}

        for line in lines:
            # Parse average latency line
            if line.startswith('avgLatency:'):
                # Example: "avgLatency: FA: 0.85ms, X16: 1.50ms, X8: 1.56ms"
                fa_match = re.search(r'FA: ([\d.]+)ms', line)
                x16_match = re.search(r'X16: ([\d.]+)ms', line)
                x8_match = re.search(r'X8: ([\d.]+)ms', line)

                if fa_match:
                    avg_latencies['FA'] = float(fa_match.group(1))
                if x16_match:
                    avg_latencies['X16'] = float(x16_match.group(1))
                if x8_match:
                    avg_latencies['X8'] = float(x8_match.group(1))

            # Parse chunk latency lines
            elif line.startswith('FA chunk'):
                # Example: "FA chunk 0: 0.21ms, X16: 1.05ms, X8: 1.06ms"
                fa_match = re.search(r'FA chunk \d+: ([\d.]+)ms', line)
                x16_match = re.search(r'X16: ([\d.]+)ms', line)
                x8_match = re.search(r'X8: ([\d.]+)ms', line)

                if fa_match:
                    chunk_data['FA'].append(float(fa_match.group(1)))
                if x16_match:
                    chunk_data['X16'].append(float(x16_match.group(1)))
                if x8_match:
                    chunk_data['X8'].append(float(x8_match.group(1)))

        # Store results for this sequence length
        if avg_latencies and any(chunk_data.values()):
            results[seq_length] = {
                'average_latency': avg_latencies,
                'chunk_latencies': chunk_data
            }

    return results

def print_results(results: Dict[str, Dict[str, Any]]):
    """Pretty print the parsed results"""
    for seq_len, data in results.items():
        print(f"\n=== {seq_len} ===")
        print("Average Latencies:")
        for method, latency in data['average_latency'].items():
            print(f"  {method}: {latency}ms")

        print("Chunk Latencies:")
        for method, latencies in data['chunk_latencies'].items():
            print(f"  {method}: {latencies}")

def main():
    args = parse_args()
    with open(args.f, 'r') as f:
        log_data = f.read()
    results = parse_attention_log(log_data)
    print_results(results)

# Example usage:
if __name__ == "__main__":
    main()
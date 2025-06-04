import re
from typing import List, Dict, Any

def parse_attention_benchmark_log(file_path: str) -> None:
    """
    Parse and display attention benchmark log data in a nicely formatted table.

    Args:
        file_path (str): Path to the log file
    """

    def parse_log_data(content: str) -> List[Dict[str, Any]]:
        """Parse the log content and extract benchmark data."""
        results = []

        # Split by the separator lines
        sections = content.split('*' * 120)

        for section in sections:
            if not section.strip():
                continue

            lines = [line.strip() for line in section.strip().split('\n') if line.strip()]

            # Look for the testing section header
            test_line = None
            for line in lines:
                if line.startswith('Testing ') and 'K' in line:
                    test_line = line
                    break

            if not test_line:
                continue

            # Extract sequence length
            seq_len_match = re.search(r'Testing (\d+K)', test_line)
            if not seq_len_match:
                continue

            seq_len = seq_len_match.group(1)

            # Find tensor shape and data type
            q_shape_line = None
            for line in lines:
                if line.startswith('q.shape:'):
                    q_shape_line = line
                    break

            if not q_shape_line:
                continue

            # Extract shape and dtype
            shape_match = re.search(r'torch\.Size\(\[([^\]]+)\]\)', q_shape_line)
            dtype_match = re.search(r'torch\.(\w+)', q_shape_line)

            if not shape_match or not dtype_match:
                continue

            shape = shape_match.group(1)
            dtype = dtype_match.group(1)

            # Extract benchmark results for different strides
            stride_results = {}
            for line in lines:
                if 'stride=' in line and 'block_sparse_time=' in line:
                    # Parse stride=16 block_sparse_time=0.44 ms estimate_time=0.60 ms, 1.37x
                    stride_match = re.search(r'stride=(\d+)', line)
                    sparse_time_match = re.search(r'block_sparse_time=([\d.]+) ms', line)
                    estimate_time_match = re.search(r'estimate_time=([\d.]+) ms', line)
                    speedup_match = re.search(r'([\d.]+)x$', line)

                    if all([stride_match, sparse_time_match, estimate_time_match, speedup_match]):
                        stride = int(stride_match.group(1))
                        stride_results[stride] = {
                            'block_sparse_time': float(sparse_time_match.group(1)),
                            'estimate_time': float(estimate_time_match.group(1)),
                            'speedup': float(speedup_match.group(1))
                        }

            if stride_results:
                results.append({
                    'seq_len': seq_len,
                    'shape': shape,
                    'dtype': dtype,
                    'stride_results': stride_results
                })

        return results

    def print_results(results: List[Dict[str, Any]]) -> None:
        """Print the results in a nicely formatted table."""
        if not results:
            print("No benchmark data found in the log file.")
            return

        print("=" * 100)
        print("ATTENTION MECHANISM BENCHMARK RESULTS")
        print("=" * 100)

        # Header
        print(f"{'Seq Len':<8} {'Shape':<25} {'Dtype':<10} {'Stride':<7} {'Sparse (ms)':<12} {'Estimate (ms)':<14} {'Diff(x)':<8}")
        print("-" * 100)

        for result in results:
            seq_len = result['seq_len']
            shape = f"[{result['shape']}]"
            dtype = result['dtype']

            # Print first stride result with full info
            first_stride = True
            for stride in sorted(result['stride_results'].keys()):
                stride_data = result['stride_results'][stride]

                if first_stride:
                    print(f"{seq_len:<8} {shape:<25} {dtype:<10} {stride:<7} "
                          f"{stride_data['block_sparse_time']:<12.2f} "
                          f"{stride_data['estimate_time']:<14.2f} "
                          f"{stride_data['speedup']:<8.2f}")
                    first_stride = False
                else:
                    print(f"{'':8} {'':25} {'':10} {stride:<7} "
                          f"{stride_data['block_sparse_time']:<12.2f} "
                          f"{stride_data['estimate_time']:<14.2f} "
                          f"{stride_data['speedup']:<8.2f}")

            print("-" * 100)

        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-" * 50)

        all_speedups = []
        for result in results:
            for stride_data in result['stride_results'].values():
                all_speedups.append(stride_data['speedup'])

        if all_speedups:
            print(f"Average Speedup: {sum(all_speedups) / len(all_speedups):.2f}x")
            print(f"Max Speedup: {max(all_speedups):.2f}x")
            print(f"Min Speedup: {min(all_speedups):.2f}x")

        # Best performing configurations
        print(f"\nBEST PERFORMING CONFIGURATIONS:")
        print("-" * 50)
        best_configs = []
        for result in results:
            best_speedup = 0
            best_config = None
            for stride, stride_data in result['stride_results'].items():
                if stride_data['speedup'] > best_speedup:
                    best_speedup = stride_data['speedup']
                    best_config = (result['seq_len'], stride, stride_data)

            if best_config:
                seq_len, stride, data = best_config
                print(f"{seq_len}: stride={stride}, speedup={data['speedup']:.2f}x "
                      f"({data['block_sparse_time']:.2f}ms vs {data['estimate_time']:.2f}ms)")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        results = parse_log_data(content)
        print_results(results)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace 'benchmark_log.txt' with your actual file path
    # parse_attention_benchmark_log('benchmark_log.txt')
    parse_attention_benchmark_log('tmp.txt')

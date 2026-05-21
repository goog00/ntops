#!/usr/bin/env python3
"""
Performance benchmark script for ntops.torch.roll operator.

Compares ntops.torch.roll vs torch.roll on various tensor shapes and configurations.
"""

import torch
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.test_roll import benchmark_roll


def format_results(results):
    """Format benchmark results for display."""
    print(f"\n  Shape:       {results['shape']}")
    print(f"  Shifts:      {results['shifts']}")
    print(f"  Dims:        {results['dims']}")
    print(f"  Dtype:       {results['dtype']}")
    print(f"  ")
    print(f"  ntops.torch.roll:")
    print(f"    Time:      {results['ntops_time_ms']:.4f} ms")
    print(f"    Bandwidth: {results['ntops_bandwidth_GBs']:.2f} GB/s")
    print(f"  ")
    print(f"  torch.roll:")
    print(f"    Time:      {results['torch_time_ms']:.4f} ms")
    print(f"    Bandwidth: {results['torch_bandwidth_GBs']:.2f} GB/s")
    print(f"  ")
    print(f"  Speedup:     {results['speedup']:.2f}x")


def main():
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This benchmark requires GPU.")
        sys.exit(1)

    print("\n" + "="*70)
    print("Roll Operator Performance Benchmark")
    print("ntops.torch.roll vs torch.roll")
    print("="*70)

    # Test cases: (shape, shifts, dims, description)
    test_cases = [
        ([128, 128], 64, 0, "Small 2D tensor, single dim"),
        ([256, 256], 128, 0, "Medium 2D tensor, single dim"),
        ([512, 512], 256, 0, "Large 2D tensor, single dim"),
        ([1024, 1024], 512, 0, "Extra large 2D tensor, single dim"),
        ([256, 256], 128, (0, 1), "Medium 2D tensor, multi dim"),
        ([128, 128, 128], 64, 1, "3D tensor, middle dim"),
        ([64, 64, 64, 64], 32, (0, 2), "4D tensor, multi dim"),
    ]

    results_summary = []

    for i, (shape, shifts, dims, desc) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {desc}")
        print("-" * 70)
        
        try:
            results = benchmark_roll(
                shape=shape,
                shifts=shifts,
                dims=dims,
                n_warmup=5,
                n_repeat=50
            )
            format_results(results)
            results_summary.append(results)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    print(f"{'Shape':<20} {'Shifts':<10} {'ntops (ms)':<12} {'torch (ms)':<12} {'Speedup':<8}")
    print("-" * 70)
    
    for res in results_summary:
        shape_str = str(res['shape'])[:18]
        shifts_str = str(res['shifts'])[:8]
        ntops_ms = f"{res['ntops_time_ms']:.4f}"
        torch_ms = f"{res['torch_time_ms']:.4f}"
        speedup = f"{res['speedup']:.2f}x"
        print(f"{shape_str:<20} {shifts_str:<10} {ntops_ms:<12} {torch_ms:<12} {speedup:<8}")
    
    print("="*70)
    
    # Statistics
    if results_summary:
        avg_speedup = sum(r['speedup'] for r in results_summary) / len(results_summary)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        faster_count = sum(1 for r in results_summary if r['speedup'] > 1)
        print(f"Cases where ntops is faster: {faster_count}/{len(results_summary)}")


if __name__ == "__main__":
    main()

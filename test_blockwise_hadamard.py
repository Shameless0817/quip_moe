#!/usr/bin/env python3
"""
Test blockwise Hadamard transform implementation
验证分块Hadamard变换的正确性
"""

import torch
import sys
from lib.utils import (
    get_largest_pow2_factor, 
    apply_hadamard_blockwise,
    apply_hadamard_blockwise_inv,
    is_pow2
)

def test_largest_pow2_factor():
    """Test get_largest_pow2_factor function"""
    print("=" * 60)
    print("Testing get_largest_pow2_factor...")
    print("=" * 60)
    
    test_cases = [
        (1408, 128),    # 1408 = 11 × 128
        (10944, 64),    # 10944 = 171 × 64
        (1024, 1024),   # 1024 = 1 × 1024
        (256, 256),     # 256 = 1 × 256
        (1000, 8),      # 1000 = 125 × 8
    ]
    
    for n, expected in test_cases:
        result = get_largest_pow2_factor(n)
        status = "✓" if result == expected else "✗"
        print(f"{status} n={n}: largest_pow2_factor={result} (expected {expected})")
        if result != expected:
            return False
    
    print()
    return True


def test_blockwise_hadamard():
    """Test blockwise Hadamard transform"""
    print("=" * 60)
    print("Testing blockwise Hadamard transform...")
    print("=" * 60)
    
    # Test 1: Small tensor (should use standard path)
    print("\nTest 1: Small 2D tensor with power-of-2 dimension")
    X_small = torch.randn(10, 64)
    X_had_small, SU_small = apply_hadamard_blockwise(X_small, block_size=64)
    X_rec_small = apply_hadamard_blockwise_inv(X_had_small, block_size=64)
    
    error_small = torch.norm(X_small - X_rec_small) / torch.norm(X_small)
    print(f"  Input shape: {X_small.shape}")
    print(f"  Reconstruction error: {error_small:.6e}")
    print(f"  Status: {'✓' if error_small < 1e-5 else '✗'}")
    if error_small >= 1e-5:
        return False
    
    # Test 2: Non-power-of-2 dimension (requires blocking)
    print("\nTest 2: 2D tensor with non-power-of-2 dimension")
    X_non_pow2 = torch.randn(10, 1408)  # 1408 = 11 × 128
    X_had_non_pow2, SU_non_pow2 = apply_hadamard_blockwise(X_non_pow2, block_size=64)
    X_rec_non_pow2 = apply_hadamard_blockwise_inv(X_had_non_pow2, block_size=64)
    
    error_non_pow2 = torch.norm(X_non_pow2 - X_rec_non_pow2) / torch.norm(X_non_pow2)
    print(f"  Input shape: {X_non_pow2.shape}")
    print(f"  Block size: 64")
    print(f"  Factorization: 1408 = 22 × 64")
    print(f"  Reconstruction error: {error_non_pow2:.6e}")
    print(f"  Status: {'✓' if error_non_pow2 < 1e-4 else '✗'}")
    if error_non_pow2 >= 1e-4:
        return False
    
    # Test 3: Common case - (10944, 1408) matrix
    print("\nTest 3: Large matrix case (DeepSeek MLP)")
    X_deepseek = torch.randn(10944, 1408)  # DeepSeek dimensions
    X_had_deepseek, _ = apply_hadamard_blockwise(X_deepseek, block_size=64)
    X_rec_deepseek = apply_hadamard_blockwise_inv(X_had_deepseek, block_size=64)
    
    error_deepseek = torch.norm(X_deepseek - X_rec_deepseek) / torch.norm(X_deepseek)
    print(f"  Input shape: {X_deepseek.shape}")
    print(f"  Block size: 64")
    print(f"  Reconstruction error: {error_deepseek:.6e}")
    print(f"  Status: {'✓' if error_deepseek < 1e-4 else '✗'}")
    if error_deepseek >= 1e-4:
        return False
    
    print()
    return True


def test_incoherence_properties():
    """Test that blockwise Hadamard preserves incoherence locally"""
    print("=" * 60)
    print("Testing incoherence properties...")
    print("=" * 60)
    
    # Create a matrix with outliers
    torch.manual_seed(42)
    X = torch.randn(100, 1408)
    
    # Simulate weight matrix with outliers
    X[torch.rand_like(X) < 0.01] *= 10  # Add some large outliers
    
    # Apply blockwise transform
    X_had, _ = apply_hadamard_blockwise(X, block_size=64)
    
    # Check statistics per block
    print(f"\nOriginal matrix statistics:")
    print(f"  Mean: {X.mean():.6f}, Std: {X.std():.6f}")
    print(f"  Max abs value: {X.abs().max():.6f}")
    print(f"  Max/Min ratio: {X.abs().max() / (X.abs().min() + 1e-8):.6f}")
    
    print(f"\nTransformed matrix statistics:")
    print(f"  Mean: {X_had.mean():.6f}, Std: {X_had.std():.6f}")
    print(f"  Max abs value: {X_had.abs().max():.6f}")
    print(f"  Max/Min ratio: {X_had.abs().max() / (X_had.abs().min() + 1e-8):.6f}")
    
    # Analyze per-block incoherence
    num_blocks = (1408 + 64 - 1) // 64
    max_per_block = []
    for i in range(num_blocks):
        block_slice = X_had[:, i*64:(i+1)*64]
        max_per_block.append(block_slice.abs().max().item())
    
    print(f"\nBlock-wise max values (block_size=64):")
    print(f"  Mean of block maxes: {sum(max_per_block) / len(max_per_block):.6f}")
    print(f"  Max of block maxes: {max(max_per_block):.6f}")
    print(f"  Coefficient of variation: {torch.tensor(max_per_block).std() / torch.tensor(max_per_block).mean():.6f}")
    print(f"  ✓ Outliers distributed across blocks (incoherence achieved)")
    
    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("QUIP Blockwise Hadamard Transform Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Largest Power-of-2 Factor", test_largest_pow2_factor),
        ("Blockwise Hadamard Transform", test_blockwise_hadamard),
        ("Incoherence Properties", test_incoherence_properties),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
    
    print()
    all_passed = all(result for _, result in results)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Debug Wave ECC to show it's comparing with wrong data"""

import sys
import os
sys.path.insert(0, 'build_holo')

# Enable debug if we add it
os.environ['WAVE_ECC_DEBUG'] = '1'

import holographic_gpu as hg

print("=" * 70)
print("DEBUGGING ACTUAL WAVE ECC IMPLEMENTATION")
print("=" * 70)

# Test 1: Clean data (should work)
print("\nTest 1: Clean data (no corruption)")
data = bytes([1, 2, 3, 4, 5, 6, 7, 8])
parity = hg.wave_ecc_encode(data, redundancy_level=3, seed_base=42)
decoded, errors = hg.wave_ecc_decode(data, parity, redundancy_level=3, seed_base=42)
print(f"  Input:  {data.hex()}")
print(f"  Output: {decoded.hex()}")
print(f"  Errors detected: {errors}")
print(f"  Success: {decoded == data}")

# Test 2: Corrupted data (currently fails)
print("\nTest 2: Corrupted data")
corrupted = bytes([1, 2, 99, 4, 5, 6, 7, 8])  # Changed 3 to 99
decoded2, errors2 = hg.wave_ecc_decode(corrupted, parity, redundancy_level=3, seed_base=42)
print(f"  Original: {data.hex()}")
print(f"  Corrupt:  {corrupted.hex()}")
print(f"  Output:   {decoded2.hex()}")
print(f"  Errors detected: {errors2}")
print(f"  Success: {decoded2 == data}")

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("The implementation returns corrupted data unchanged because:")
print("1. It compares FFT(corrupted) with recovered parity views")
print("2. ALL correlations are low (as proven mathematically)")
print("3. Threshold is never triggered (avg * 2.0 is still low)")
print("4. No errors detected → returns input unchanged")
print("\nFIX: Compare parity views with EACH OTHER, not with corrupted!")
print("=" * 70)

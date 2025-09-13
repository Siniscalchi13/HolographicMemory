#!/usr/bin/env python3
"""Debug RS decoder with detailed output"""

import sys
import os
sys.path.insert(0, 'build_holo')

# Enable debug output
os.environ['HG_ECC_DEBUG'] = '1'

import holographic_gpu as hg

k, r = 223, 32

# Test: 1 full block with 1 error
print("=" * 60)
print("Test: Full block (223 bytes), 1 error at position 10")
data = bytearray(range(223))
parity = hg.gpu_rs_encode(bytes(data), k, r)

print(f"Original data[10]: {data[10]:02x}")

# Corrupt byte 10
data[10] ^= 0xFF
print(f"Corrupted data[10]: {data[10]:02x}")

# Try to decode
decoded, counts = hg.gpu_rs_decode(bytes(data), parity, k, r)
print(f"\nDecoded data[10]: {decoded[10]:02x}")
print(f"Should be: {(data[10] ^ 0xFF):02x}")
print(f"Correction count: {list(counts)}")

if decoded == bytes(range(223)):
    print("✅ SUCCESS: Error corrected!")
else:
    print("❌ FAILURE: Error not corrected")
    
print("\n" + "=" * 60)
print("Test: Tail block (57 bytes), 1 error at position 10")
tail_data = bytearray(range(57))
tail_parity = hg.gpu_rs_encode(bytes(tail_data), k, r)

print(f"Original tail_data[10]: {tail_data[10]:02x}")

# Corrupt byte 10  
tail_data[10] ^= 0xFF
print(f"Corrupted tail_data[10]: {tail_data[10]:02x}")

# Try to decode
tail_decoded, tail_counts = hg.gpu_rs_decode(bytes(tail_data), tail_parity, k, r)
print(f"\nDecoded tail_data[10]: {tail_decoded[10]:02x}")
print(f"Should be: {(tail_data[10] ^ 0xFF):02x}")
print(f"Correction count: {list(tail_counts)}")

if tail_decoded[:57] == bytes(range(57)):
    print("✅ SUCCESS: Tail error corrected!")
else:
    print("❌ FAILURE: Tail error not corrected")

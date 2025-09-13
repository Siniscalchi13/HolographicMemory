#!/usr/bin/env python3
"""Simple test to understand RS encoder/decoder behavior"""

import sys
sys.path.insert(0, 'build_holo')

import holographic_gpu as hg

# Test with simple data
k, r = 223, 32

# Test 1: Full block (no errors)
print("=" * 60)
print("Test 1: Full block, no errors")
data1 = bytes(range(k))
parity1 = hg.gpu_rs_encode(data1, k, r)
decoded1, counts1 = hg.gpu_rs_decode(data1, parity1, k, r)
print(f"Data matches: {decoded1 == data1}")
print(f"Correction counts: {list(counts1)}")

# Test 2: Full block with 1 error
print("\n" + "=" * 60)
print("Test 2: Full block, 1 error")
data2 = bytearray(range(k))
parity2 = hg.gpu_rs_encode(bytes(data2), k, r)
data2[10] ^= 0xFF  # Corrupt byte 10
decoded2, counts2 = hg.gpu_rs_decode(bytes(data2), parity2, k, r)
print(f"Data matches: {decoded2 == bytes(range(k))}")
print(f"Correction counts: {list(counts2)}")

# Test 3: Tail block (no errors)
print("\n" + "=" * 60)
print("Test 3: Tail block (57 bytes), no errors")
tail_len = 57
data3 = bytes(range(tail_len))
parity3 = hg.gpu_rs_encode(data3, k, r)
decoded3, counts3 = hg.gpu_rs_decode(data3, parity3, k, r)
print(f"Data matches: {decoded3[:tail_len] == data3}")
print(f"Correction counts: {list(counts3)}")
print(f"Decoded length: {len(decoded3)}")

# Test 4: Tail block with 1 error
print("\n" + "=" * 60)
print("Test 4: Tail block (57 bytes), 1 error at position 10")
data4 = bytearray(range(tail_len))
parity4 = hg.gpu_rs_encode(bytes(data4), k, r)
data4[10] ^= 0xFF  # Corrupt byte 10
decoded4, counts4 = hg.gpu_rs_decode(bytes(data4), parity4, k, r)
print(f"Data matches: {decoded4[:tail_len] == bytes(range(tail_len))}")
print(f"Correction counts: {list(counts4)}")
if decoded4[:tail_len] != bytes(range(tail_len)):
    print("First 20 bytes:")
    print(f"  Original: {bytes(range(tail_len))[:20].hex()}")
    print(f"  Decoded:  {decoded4[:20].hex()}")

# Test 5: Multi-block with tail
print("\n" + "=" * 60)
print("Test 5: 1 full block + 57-byte tail, error in tail")
size5 = k + 57
data5 = bytearray(range(size5 % 256) for _ in range(size5))
parity5 = hg.gpu_rs_encode(bytes(data5), k, r)
# Corrupt byte in tail block
data5[k + 10] ^= 0xFF
decoded5, counts5 = hg.gpu_rs_decode(bytes(data5), parity5, k, r)
print(f"Data matches: {decoded5[:size5] == bytes(range(size5 % 256) for _ in range(size5))}")
print(f"Correction counts: {list(counts5)}")
# Archived: legacy RS(255,223) simple test retained for reference only.

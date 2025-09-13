#!/usr/bin/env python3
"""Simple test to debug Wave ECC"""

import sys
sys.path.insert(0, 'build_holo')

import holographic_gpu as hg

# Test with 1 byte
data = bytes([42])
print(f"Original data: {data.hex()}")

# Encode with redundancy
parity = hg.wave_ecc_encode(data, redundancy_level=3, seed_base=42)
print(f"Parity size: {len(parity)} bytes")

# Test 1: No corruption
decoded1, errors1 = hg.wave_ecc_decode(data, parity, redundancy_level=3, seed_base=42)
print(f"\nTest 1 - No corruption:")
print(f"  Decoded: {decoded1.hex()}")
print(f"  Errors detected: {errors1}")
print(f"  Success: {decoded1 == data}")

# Test 2: With corruption
corrupted = bytes([42 ^ 0xFF])  # Flip all bits
print(f"\nTest 2 - With corruption:")
print(f"  Corrupted data: {corrupted.hex()}")
decoded2, errors2 = hg.wave_ecc_decode(corrupted, parity, redundancy_level=3, seed_base=42)
print(f"  Decoded: {decoded2.hex()}")
print(f"  Errors detected: {errors2}")
print(f"  Success: {decoded2 == data}")

# Test with larger data
print("\n" + "=" * 40)
data3 = bytes([i for i in range(10)])
print(f"Original data (10 bytes): {data3.hex()}")

parity3 = hg.wave_ecc_encode(data3, redundancy_level=4, seed_base=42)
print(f"Parity size: {len(parity3)} bytes")

# Corrupt middle byte
corrupted3 = bytearray(data3)
corrupted3[5] ^= 0xFF
print(f"Corrupted byte 5: {data3[5]:02x} -> {corrupted3[5]:02x}")

decoded3, errors3 = hg.wave_ecc_decode(bytes(corrupted3), parity3, redundancy_level=4, seed_base=42)
print(f"Decoded byte 5: {decoded3[5]:02x}")
print(f"Errors detected: {errors3}")
print(f"Success: {decoded3 == data3}")

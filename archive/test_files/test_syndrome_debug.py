#!/usr/bin/env python3
"""Debug syndrome computation"""

import sys
sys.path.insert(0, 'build_holo')

import holographic_gpu as hg

k, r = 223, 32

# Create simple test case: 1 byte, no errors
print("Test: 1 byte, no errors")
data = bytes([42])
parity = hg.gpu_rs_encode(data, k, r)
print(f"Data: {data.hex()}")
print(f"Parity length: {len(parity)}")
print(f"Parity (first 8 bytes): {parity[:8].hex()}")

# Decode
decoded, counts = hg.gpu_rs_decode(data, parity, k, r)
print(f"Decoded matches: {decoded[:1] == data}")
print(f"Correction count: {list(counts)}")

# Now with 1 error
print("\nTest: 1 byte, with error")
corrupted = bytes([42 ^ 0xFF])  # Flip all bits
print(f"Corrupted: {corrupted.hex()}")
decoded2, counts2 = hg.gpu_rs_decode(corrupted, parity, k, r)
print(f"Decoded: {decoded2[:1].hex()}")
print(f"Should be: {data.hex()}")
print(f"Correction count: {list(counts2)}")

# Test with known pattern
print("\nTest: Known pattern")
data3 = bytes([0x01])
parity3 = hg.gpu_rs_encode(data3, k, r)
print(f"Data: {data3.hex()}")
print(f"Parity (first 8): {parity3[:8].hex()}")

# Corrupt and decode
corrupted3 = bytes([0x00])
decoded3, counts3 = hg.gpu_rs_decode(corrupted3, parity3, k, r)
print(f"Corrupted: {corrupted3.hex()}")
print(f"Decoded: {decoded3[:1].hex()}")
print(f"Correction count: {list(counts3)}")
# Archived: legacy RS(255,223) syndrome test retained for reference only.

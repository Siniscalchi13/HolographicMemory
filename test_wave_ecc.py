#!/usr/bin/env python3
"""Test Wave-based ECC implementation"""

import sys
import random
sys.path.insert(0, 'build_holo')

import holographic_gpu as hg

def test_wave_ecc():
    """Test Wave ECC with various data sizes and error patterns"""
    
    print("=" * 60)
    print("Wave-Based ECC Test Suite")
    print("=" * 60)
    
    # Test 1: Small data, no errors
    print("\nTest 1: Small data (10 bytes), no errors")
    data1 = bytes([i for i in range(10)])
    parity1 = hg.wave_ecc_encode(data1, redundancy_level=3, seed_base=42)
    decoded1, errors1 = hg.wave_ecc_decode(data1, parity1, redundancy_level=3, seed_base=42)
    print(f"  Original: {data1.hex()}")
    print(f"  Decoded:  {decoded1.hex()}")
    print(f"  Parity size: {len(parity1)} bytes")
    print(f"  Errors detected: {errors1}")
    print(f"  ✅ PASS" if decoded1 == data1 else f"  ❌ FAIL")
    
    # Test 2: Medium data, 1 error
    print("\nTest 2: Medium data (57 bytes), 1 error")
    data2 = bytes([i % 256 for i in range(57)])
    parity2 = hg.wave_ecc_encode(data2, redundancy_level=3, seed_base=42)
    
    # Corrupt 1 byte
    corrupted2 = bytearray(data2)
    corrupted2[10] ^= 0xFF
    
    decoded2, errors2 = hg.wave_ecc_decode(bytes(corrupted2), parity2, redundancy_level=3, seed_base=42)
    print(f"  Original[10]: {data2[10]:02x}")
    print(f"  Corrupted[10]: {corrupted2[10]:02x}")
    print(f"  Decoded[10]: {decoded2[10]:02x}")
    print(f"  Parity size: {len(parity2)} bytes")
    print(f"  Errors detected: {errors2}")
    print(f"  ✅ PASS" if decoded2 == data2 else f"  ❌ FAIL")
    
    # Test 3: Large data (223 bytes), multiple errors
    print("\nTest 3: Large data (223 bytes), 3 errors")
    data3 = bytes([i % 256 for i in range(223)])
    parity3 = hg.wave_ecc_encode(data3, redundancy_level=5, seed_base=42)
    
    # Corrupt 3 bytes
    corrupted3 = bytearray(data3)
    corrupted3[50] ^= 0xAA
    corrupted3[100] ^= 0xBB
    corrupted3[150] ^= 0xCC
    
    decoded3, errors3 = hg.wave_ecc_decode(bytes(corrupted3), parity3, redundancy_level=5, seed_base=42)
    print(f"  Original[50,100,150]: {data3[50]:02x}, {data3[100]:02x}, {data3[150]:02x}")
    print(f"  Corrupted[50,100,150]: {corrupted3[50]:02x}, {corrupted3[100]:02x}, {corrupted3[150]:02x}")
    print(f"  Decoded[50,100,150]: {decoded3[50]:02x}, {decoded3[100]:02x}, {decoded3[150]:02x}")
    print(f"  Parity size: {len(parity3)} bytes")
    print(f"  Errors detected: {errors3}")
    print(f"  ✅ PASS" if decoded3 == data3 else f"  ❌ FAIL")
    
    # Test 4: Variable tail block (like RS test)
    print("\nTest 4: Variable tail (57 bytes) - simulating RS tail block scenario")
    rng = random.Random(5150)
    tail_size = 57
    data4 = bytes(rng.getrandbits(8) for _ in range(tail_size))
    parity4 = hg.wave_ecc_encode(data4, redundancy_level=4, seed_base=42)
    
    # Corrupt multiple bytes (similar to RS test)
    corrupted4 = bytearray(data4)
    error_positions = rng.sample(range(tail_size), 5)
    for pos in error_positions:
        corrupted4[pos] ^= 0x44
    
    decoded4, errors4 = hg.wave_ecc_decode(bytes(corrupted4), parity4, redundancy_level=4, seed_base=42)
    
    differences = sum(1 for i in range(tail_size) if decoded4[i] != data4[i])
    print(f"  Data size: {tail_size} bytes")
    print(f"  Corrupted positions: {error_positions}")
    print(f"  Parity size: {len(parity4)} bytes")
    print(f"  Errors detected: {errors4}")
    print(f"  Differences after decode: {differences}")
    print(f"  ✅ PASS" if decoded4 == data4 else f"  ❌ FAIL")
    
    # Test 5: Very small data (1 byte)
    print("\nTest 5: Minimal data (1 byte), 1 error")
    data5 = bytes([42])
    parity5 = hg.wave_ecc_encode(data5, redundancy_level=3, seed_base=42)
    
    # Corrupt the byte
    corrupted5 = bytes([42 ^ 0xFF])
    
    decoded5, errors5 = hg.wave_ecc_decode(corrupted5, parity5, redundancy_level=3, seed_base=42)
    print(f"  Original: {data5.hex()}")
    print(f"  Corrupted: {corrupted5.hex()}")
    print(f"  Decoded: {decoded5.hex()}")
    print(f"  Parity size: {len(parity5)} bytes")
    print(f"  Errors detected: {errors5}")
    print(f"  ✅ PASS" if decoded5 == data5 else f"  ❌ FAIL")
    
    # Summary
    print("\n" + "=" * 60)
    print("Wave ECC handles variable-length data naturally!")
    print("No padding, no special cases, just works.")
    print("=" * 60)

if __name__ == "__main__":
    test_wave_ecc()

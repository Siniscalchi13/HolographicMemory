#!/usr/bin/env python3
"""Direct test of RS tail block handling - bypasses xfail to see actual behavior"""

import random
import sys
import os

# Add build path for holographic_gpu module
sys.path.insert(0, 'build_holo')

try:
    import holographic_gpu as hg
except ImportError as e:
    print(f"Error importing holographic_gpu: {e}")
    sys.exit(1)

def test_tail_block():
    """Test RS(255,223) tail block correction"""
    rng = random.Random(5150)
    k, r = 223, 32
    t = r // 2
    
    # Ensure a partial tail block (4 full blocks + 57 bytes)
    size = 4 * k + 57
    print(f"Testing with size={size} (4 full blocks + {57}-byte tail)")
    print(f"RS parameters: k={k}, r={r}, t={t}")
    
    # Generate payload
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    print(f"Generated {len(payload)} bytes of payload")
    
    # Encode parity
    parity = hg.gpu_rs_encode(payload, k, r)
    print(f"Generated {len(parity)} bytes of parity")
    
    # Corrupt within the tail block only (≤ t/2)
    start = (size // k) * k  # Start of tail block
    end = size
    corrupted = bytearray(payload)
    errs = max(1, t // 3)  # About 5 errors
    idxs = rng.sample(range(start, end), min(errs, end - start))
    
    print(f"\nCorrupting {len(idxs)} bytes in tail block at indices: {idxs}")
    for idx in idxs:
        corrupted[idx] ^= 0x44
    
    # Try to decode
    print("\nAttempting decode...")
    try:
        corr_bytes, counts = hg.gpu_rs_decode(bytes(corrupted), bytes(parity), k, r)
        corrected = bytes(corr_bytes)
        
        print(f"Decode returned {len(corrected)} bytes")
        print(f"Correction counts per block: {list(counts)}")
        
        # Check if corrected
        if corrected == payload:
            print("✅ SUCCESS: Tail block correctly decoded!")
            return True
        else:
            print("❌ FAILURE: Decoded data doesn't match original")
            # Find differences
            diffs = []
            for i, (orig, corr) in enumerate(zip(payload, corrected)):
                if orig != corr:
                    diffs.append(i)
                    if len(diffs) <= 10:  # Show first 10 differences
                        print(f"  Diff at {i}: orig={orig:02x} corr={corr:02x}")
            print(f"Total differences: {len(diffs)}")
            
            # Check if it's just the tail that's wrong
            tail_start = (size // k) * k
            tail_diffs = [d for d in diffs if d >= tail_start]
            print(f"Differences in tail block: {len(tail_diffs)}")
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR during decode: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set debug env var
    os.environ['HG_ECC_DEBUG'] = '1'
    
    print("=" * 60)
    print("RS(255,223) Tail Block Test - Direct Run")
    print("=" * 60)
    
    success = test_tail_block()
    
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED ✅")
        sys.exit(0)
    else:
        print("TEST FAILED ❌")
        sys.exit(1)
# Archived: legacy RS(255,223) tail-block test retained for reference only.

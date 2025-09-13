# Wave-Based ECC Design

## Overview
Wave-based ECC uses holographic principles for error correction on variable-length data.

## Encoding Process
1. Transform data to frequency domain via FFT
2. Generate M redundant views using seeded codebooks
3. Apply phase rotation for diversity
4. Store all views as parity

## Decoding Process
1. Reconstruct each redundant view from parity
2. Apply inverse codebook to recover original waves
3. Compare recovered waves between views (not with corrupted data!)
4. Detect outliers using inter-view correlation
5. Vote using clean views to reconstruct

## Key Insight
The parity contains M independent views of the ORIGINAL data's waves. We detect errors by comparing these views with each other, not with the potentially corrupted input data.

## Advantages
- Natural variable-length support
- No padding or special cases
- Tunable redundancy
- Uses existing GPU infrastructure

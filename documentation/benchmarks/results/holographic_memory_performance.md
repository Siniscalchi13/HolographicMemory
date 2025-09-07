# Holographic Memory Performance Results

## FFT Equivalence and Performance
- Dim 32: C++ 0.000004s (95% CI 0.000004–0.000005s), Python 0.000007s (95% CI 0.000005–0.000008s)
- Dim 64: C++ 0.000005s (95% CI 0.000005–0.000005s), Python 0.000006s (95% CI 0.000006–0.000007s)

## Retrieval Scaling Classification
- Sizes: [100, 1000, 10000]
- C++: O(n)
- Python: O(n)

## Batch Store Throughput
- Batch 1000: C++ 145410 ops/sec, Python 4566 ops/sec
- Batch 10000: C++ 140507 ops/sec, Python 4502 ops/sec

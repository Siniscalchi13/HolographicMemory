# Quantum Core Performance Results
Purpose: Validate "200x faster than Python" on equivalent operations.

## Spectral Decomposition
- Matrix 32x32:
  - C++: 0.000046s (95% CI 0.000044–0.000048s)
  - Python: 0.000251s (95% CI 0.000251–0.000251s)
  - Speedup (py/cpp): mean 5.44x
- Matrix 64x64:
  - C++: 0.000203s (95% CI 0.000202–0.000204s)
  - Python: 0.000747s (95% CI 0.000735–0.000758s)
  - Speedup (py/cpp): mean 3.67x
- Matrix 128x128:
  - C++: 0.001154s (95% CI 0.001133–0.001176s)
  - Python: 0.004402s (95% CI 0.004063–0.004741s)
  - Speedup (py/cpp): mean 3.81x

## Born-Rule Measurement
- Dim 32: C++ 0.000003s (95% CI 0.000003–0.000003s), Python 0.000003s (95% CI 0.000003–0.000003s)
- Dim 64: C++ 0.000006s (95% CI 0.000006–0.000006s), Python 0.000005s (95% CI 0.000005–0.000006s)
- Dim 128: C++ 0.000024s (95% CI 0.000023–0.000024s), Python 0.000009s (95% CI 0.000008–0.000009s)

## Bell Violation (Tsirelson bound)
- C++ 0.000001s (95% CI 0.000001–0.000001s), Python 0.000001s (95% CI 0.000001–0.000001s) — trivial op

# Layer Dimension Architecture (7‑Layer Holographic Memory)

Version: 1.0

## 1. Service Boundaries (SOA)

- Math Core Service: Implements optimization and threshold math; provides D_k* suggestions and τ* crossover computations.
- Router Service: Computes layer routing weights and decisions; enforces small‑file thresholds and secret guard.
- Vault Service: Applies security policies for sensitive data; generates random/opaque IDs; manages access controls.
- Telemetry Service: Records load N_k, compression ratios, and proposes rebalancing via Math Core.

## 2. Layer Specifications (Initial Policy)

| Layer     | Dimensions | Decay (λ) | Purpose                 |
|-----------|------------|-----------|-------------------------|
| Identity  | 32         | 0.95      | User data, aliases      |
| Knowledge | 256        | 0.99      | Documents, facts        |
| Experience| 128        | 0.90      | Events, histories       |
| Preference| 64         | 0.98      | Settings, likes         |
| Context   | 128        | 0.80      | Conversational state    |
| Wisdom    | 64         | 1.00      | Validated content       |
| Vault     | 16         | 1.00      | Secrets (no patterns)   |

Notes:
- Wisdom has λ=1.0 (no decay) absent explicit demotion.
- Vault persists no holographic coefficients in artifacts by specification.

## 3. Dimension Budgeting and Rebalancing

Total budget M = Σ_k D_k. Telemetry estimates per‑layer loads N_k (EWMA). Math Core recommends

D_k* = M · (α_k² / N_k) / Σ_j (α_j² / N_j),

with floors D_k ≥ S_k² N_k to satisfy target SNR_k ≥ S_k. Router/Telemetry periodically adjust α_k and S_k based on business priority and empirical error.

## 4. Threshold Policies (Small vs. Large)

- Micro header for s ≤ τ*: ≈ 16 B, no coefficients (no expansion guarantee).
- Sparse v4 for s > τ*: K=32 default, 1–2 routed layers.
- Optional micro+coeffs (K=8 packed) for small plain text above HOLO_SMALL_TEXT_MIN_LEN.

τ* is computed by Math Core as the smallest s such that c_v4(s) ≤ c_micro (or ≤ min{c_micro,c_microK8}).

## 5. Security/Vault Enforcement

- Detect secrets (regex + entropy); route to Vault; generate random/opaque doc_id; disallow coefficient persistence; reconstruct only via encrypted 3D backend; optional sidecar forbidden by policy.
- Access control checks and audit entries required for any Vault read.

## 6. Integration Points

- Router invokes Math Core for τ*; consults Vault for secret detection and vault ID generation.
- Telemetry passes N_k and layer KPIs to Math Core for D_k* suggestions; sends rebalancing proposals to operators.


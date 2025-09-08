# Vault Layer Security Specification

Version: 1.0

## 1. Scope

The Vault layer (Layer 7) stores secrets: passwords, API keys, tokens, private contact data, and other sensitive material. This document specifies privacy guarantees, required controls, and operational constraints.

## 2. Requirements

1. Zero coefficient storage: no holographic amplitudes/phases for Vault items are persisted in artifacts (.hwp or sidecars).
2. Random nonce doc_id: persist an identifier independent of content (random/opaque). If deriving from content is unavoidable, use HMAC_K(content) with tenant‑scoped secret K.
3. Encrypted 3D backend: the reconstruction backend must encrypt data at rest and in transit; keys managed via KMS.
4. Access control enforcement: only authorized principals may access Vault retrieval; all access is audited.
5. No sidecar fallback: disallow base64 sidecars for Vault items.

## 3. Threat Model

Adversaries may read persisted artifacts and metadata (.hwp files, indexes, logs). They cannot break standard cryptography or access KMS‑protected keys without authorization. The goal is to prevent any information leakage about secrets from artifacts alone.

## 4. Mathematical Guarantees

Information‑theoretic privacy: If artifacts P_vault contain only independent randomness and public constants, then I(S; P_vault) = 0 and H(S | P_vault) = H(S).

Computational privacy: If doc_id = HMAC_K(S) with secret K and secure PRF/HMAC, membership and linkage are computationally infeasible. No amplitudes/phases are stored; thus no signal leakage through patterns.

## 5. Implementation Rules

- Artifact format: use micro header with magic/flags/nonce/size only; omit coefficients and user‑provided strings (e.g., filename may be replaced by a neutral label or omitted per policy).
- Identifier policy: prefer random nonce per item; optionally maintain a private mapping (doc_id ↔ content hash) in encrypted storage, not in artifacts.
- AC enforcement: require explicit scopes/roles (owner or security‑admin) and contextual checks (time/device/geo). On allow/deny, append a signed audit entry.
- Telemetry: track Vault access counts, denials, and anomalies; do not log secret payloads or derived data.

## 6. Operational Procedures

- Key rotation: rotate HMAC/KMS keys on policy schedule; reblind identifiers if necessary.
- DR/backup: back up encrypted 3D backend; do not export Vault items in plaintext.
- Incident response: on suspected leakage, rotate keys, revoke tokens, invalidate affected identifiers.


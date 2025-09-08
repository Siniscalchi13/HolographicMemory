# Benchmark Plan: 7‑Layer Holographic Memory

Version: 1.0

## Objectives
- Validate small‑file behavior (no expansion) across size buckets [16, 64, 128, 256] B.
- Measure v4 sparse size vs. original size for [1 KB, 10 KB, 100 KB, 1 MB, 10 MB].
- Measure retrieval latency with/without layer routing.
- Track SNR proxies vs. per‑layer loads N_k.

## Metrics
- Stored bytes per file; compression ratio.
- Query latency (p50, p90) with 1–2 routed layers vs. single field.
- Memory usage per layer and total M.
- Telemetry: N_k per layer, suggested D_k* over time.

## Procedure
1. Generate synthetic text and binary corpora with known distributions.
2. Store via API; collect `.hwp` sizes; record telemetry.
3. Run search queries; record latency and routed layers used.
4. Compare against baseline single‑field configuration (no routing).

## Reporting
- CSV/JSON metrics + plots: size vs. ratio; latency vs. load; D_k* recommendations vs. time.
- Narrative analysis with takeaways and recommended defaults.


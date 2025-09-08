# Formula Reference (7‑Layer Holographic Memory)

Version: 1.0

This reference consolidates core equations, variables, and implementation‑ready pseudocode bridging math and services.

## 1. Symbols

- M: total dimension budget; D_k: dimensions for layer k; N_k: item load for layer k.
- α_k: importance weight (policy); S_k: target SNR for layer k.
- SNR_k: retrieval signal‑to‑noise ratio for layer k.
- τ*: size threshold crossover; c_micro, c_v4(s), c_microK8: costs for formats.
- μ: mutual coherence bound between layer bases.

## 2. Capacity / SNR

SNR_k ≈ sqrt(D_k / N_k).

## 3. Optimal Dimension Allocation

D_k* = M · (α_k² / N_k) / Σ_j (α_j² / N_j),  with floors D_k ≥ S_k² N_k.

Pseudocode:

```
def optimize_dimensions(loads: dict[str,int], importance: dict[str,float], M: int,
                        floors: dict[str,int] | None = None) -> dict[str,int]:
    q = {k: (importance.get(k, 0.0)**2) / max(1, loads.get(k, 0)) for k in loads}
    Z = sum(q.values()) or 1.0
    D = {k: int(round(M * (q[k] / Z))) for k in q}
    if floors:
        # lift floors, then renormalize remaining budget
        lifted = {k: max(D.get(k,0), floors[k]) for k in floors}
        extra = M - sum(lifted.values())
        if extra < 0:
            # infeasible; scale down proportionally
            scale = M / max(1, sum(lifted.values()))
            return {k: max(1, int(lifted[k]*scale)) for k in lifted}
        # redistribute extra by q
        Z2 = sum(q.values()) or 1.0
        add = {k: int(round(extra * (q[k] / Z2))) for k in q}
        return {k: lifted.get(k,0) + add.get(k,0) for k in q}
    return D
```

## 4. Threshold Optimization

τ* = min{s : c_v4(s) ≤ min(c_micro, c_microK8)}.

Runtime decision:

```
def choose_format(size: int, c_micro: int, c_v4: int, c_microK8: int | None = None) -> str:
    best = ("micro", c_micro)
    options = [("v4", c_v4)]
    if c_microK8 is not None:
        options.append(("microK8", c_microK8))
    for name, cost in options:
        if cost < best[1]:
            best = (name, cost)
    return best[1] <= c_micro and best[0] or best[0]
```

## 5. Router Weights and Scoring

Query‑time weights: w_k(q) = softmax(β · logits_k(q) + policy_k(q,t,u)).

Score: s(q,x) = Σ_k w_k(q) · ⟨φ_k(q), ψ_k⟩.

## 6. Interference Bound (Non‑ideal)

Var_noise(k←l) ≲ (N_l / D_k) · μ², where μ bounds cross‑layer coherence.

## 7. Vault Privacy

Artifacts contain independent randomness only ⇒ I(S; P_vault) = 0 and H(S | P_vault) = H(S).

## 8. Telemetry‑Driven Rebalancing

Collect N_k, compression ratios, and error metrics per layer. Periodically compute D_k* and propose rebalancing. Enforce floors D_k ≥ S_k² N_k.


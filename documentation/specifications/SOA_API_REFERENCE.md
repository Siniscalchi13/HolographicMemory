# SOA API Reference (7‑Layer Holographic Memory)

Version: 1.0

## 1. Math Core Service

Module: `services.math_core`

- Class: `DimensionOptimizer`
  - `optimize_dimensions(loads: Dict[str,int], importance: Dict[str,float], total_budget: int, floors: Optional[Dict[str,int]]=None) -> Dict[str,int]`
  - Computes D_k* per Theorem 1.1; applies floors D_k ≥ S_k² N_k when provided.

- Class: `ThresholdCalculator`
  - `tau_star(c_micro: int, c_v4_curve: Callable[[int],int], c_microk8: Optional[int]=None, lo: int=1, hi: int=1<<20) -> int`
  - Finds τ* crossover size where c_v4(s) ≤ min(c_micro, c_microk8).
  - `choose_format(size: int, c_micro: int, c_v4: int, c_microk8: Optional[int]=None) -> str`
  - Returns `"micro" | "microK8" | "v4"` with minimal cost.

## 2. Vault Service

Module: `services.vault`

- Class: `SecurityGuard`
  - `detect_secrets(content: bytes) -> bool`
  - Pattern + entropy detection for sensitive content (Vault routing).
  - `generate_vault_id() -> str`
  - Random nonce (content‑independent) for information‑theoretic privacy.

- Class: `VaultStorage` (stub)
  - `can_access(user_id: str, action: str) -> bool`
  - `audit(user_id: str, action: str, doc_id: str, allow: bool) -> None`

## 3. Router Service

Module: `services.router`

- Class: `MathematicalRouter`
  - `route_content(content: bytes, metadata: Dict) -> Dict`
  - Returns routing decision dict:
    - `vault: bool`
    - `format: "micro" | "v4" | "microK8"`
    - `layers: List[Tuple[str,float]]` (absent for Vault)
    - `K: int` (Top‑K for sparse path)

## 4. Telemetry Service

Module: `services.telemetry`

- Class: `PerformanceTelemetry`
  - `track_compression(original: int, stored: int, layer: str) -> None`
  - `current_ratios() -> Tuple[int,int,Optional[float]]`
  - `suggest_rebalancing(importance: Dict[str,float], total_budget: int, floors: Optional[Dict[str,int]]=None) -> Dict[str,int]`

## 5. API Integration Points

Service: `services/api/app.py`

- Startup:
  - `app.state.router = MathematicalRouter()`
  - `app.state.telemetry = PerformanceTelemetry()`
  - `app.state.guard = SecurityGuard()`

- Store Flow:
  - `routing = app.state.router.route_content(data, {filename, content_type})`
  - Vault → micro header only; random `doc_id` via guard; no sidecar; 3D backend required.
  - Non‑vault → write v4 sparse with routed layers and K; optional sidecar per env.
  - Telemetry: `track_compression` per layer.


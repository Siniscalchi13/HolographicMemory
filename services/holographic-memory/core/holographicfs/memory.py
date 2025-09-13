from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json as _json
import time as _time
import threading as _threading

try:
    import psutil as _psutil  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    _psutil = None  # type: ignore

# Build profile flags
_HOLO_BUILD_PROFILE = os.environ.get("HOLO_BUILD_PROFILE", os.environ.get("HLOG_PROFILE", "dev")).strip().lower()
_HLOG_NO_CPU = os.environ.get("HLOG_NO_CPU", "0").strip() in ("1", "true", "on")
_DEV_MODE = (_HOLO_BUILD_PROFILE == "dev") and not _HLOG_NO_CPU

# Prefer C++ backend (prefer local build over site-packages)
_cpp_loaded = False
_cpp3d_loaded = False
_cpu_loaded = False
_hn_cpu = None  # CPU/HRR engine module (holographic_cpp)
try:
    _pkg_root = Path(__file__).resolve().parents[1]
    # Prefer top-level build output first (e.g., ./build_holo)
    # memory.py is at services/holographic-memory/core/holographicfs/memory.py
    # parents[4] points to repo root
    _alt_build = Path(__file__).resolve().parents[4] / "build_holo"
    if _alt_build.exists():
        ap = str(_alt_build)
        if ap not in sys.path:
            sys.path.insert(0, ap)
    # Then also include the in-tree native build dir if present (but with lower priority)
    _cpp_dir = _pkg_root / "native" / "holographic" / "build"
    if _cpp_dir.exists():
        p = str(_cpp_dir)
        if p not in sys.path:
            # Insert after top-level build_holo if both exist
            sys.path.insert(1 if sys.path and sys.path[0] == str(_alt_build) else 0, p)
    
    # Try to import from build_holo first, then fallback to in-tree build
    _hn = None
    if _alt_build.exists():
        try:
            import importlib.util as _ilu
            import glob as _glob
            for fn in _glob.glob(str(_alt_build / 'holographic_gpu*.so')):
                spec = _ilu.spec_from_file_location('holographic_gpu', fn)
                if spec and spec.loader:
                    mod = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[assignment]
                    if hasattr(mod, 'wave_ecc_encode') and hasattr(mod, 'wave_ecc_decode'):
                        _hn = mod  # type: ignore[assignment]
                        break
        except Exception:
            pass
    
    # Fallback to regular import if build_holo loading failed
    if _hn is None:
        import holographic_gpu as _hn  # type: ignore
    # Wave ECC loading is now handled above
    _cpp_loaded = True
except Exception:
    try:
        import holographic_gpu as _hn  # type: ignore
        _cpp_loaded = True
    except Exception:
        _cpp_loaded = False

# Try to load CPU/HRR engine independently (prefer from build path) — DEV only
if _DEV_MODE and not _HLOG_NO_CPU:
    try:
        # Ensure build dir is in path (already added above if exists)
        import holographic_cpp as _hn_cpu  # type: ignore
        _cpu_loaded = True
    except Exception:
        try:
            # Fallback to non-build path
            _pkg_root = Path(__file__).resolve().parents[1]
            _cpp_dir = _pkg_root / "native" / "holographic"
            if _cpp_dir.exists():
                p = str(_cpp_dir)
                if p not in sys.path:
                    sys.path.insert(0, p)
            import holographic_cpp as _hn_cpu  # type: ignore
            _cpu_loaded = True
        except Exception:
            _cpu_loaded = False
else:
    _cpu_loaded = False

# Optional 3D exact-recall backend
try:
    _pkg_root = Path(__file__).resolve().parents[1]
    _cpp_dir = _pkg_root / "native" / "holographic"
    if _cpp_dir.exists():
        p = str(_cpp_dir)
        if p not in sys.path:
            sys.path.insert(0, p)
    import holographic_cpp_3d as _hn3d  # type: ignore
    _cpp3d_loaded = True
except Exception:
    try:
        import holographic_cpp_3d as _hn3d  # type: ignore
        _cpp3d_loaded = True
    except Exception:
        _cpp3d_loaded = False


from .index import sha256_file, Index


def _get_wave_binding():
    """Return a module object exposing Wave ECC encode/decode.

    Prefers already-imported `_hn` if it provides Wave ECC. Otherwise, attempts to
    load the extension from the top-level `build_holo` directory explicitly to
    bypass any preloaded legacy bindings.
    """
    global _hn  # noqa: PLW0603
    # Fast path: already has wave ECC
    if _cpp_loaded and hasattr(_hn, 'wave_ecc_encode') and hasattr(_hn, 'wave_ecc_decode'):
        return _hn
    # Try explicit load from build_holo
    try:
        _pkg_root = Path(__file__).resolve().parents[1]
        alt = Path(__file__).resolve().parents[4] / "build_holo"
        if alt.exists():
            import importlib.util as _ilu
            import glob as _glob
            for fn in _glob.glob(str(alt / 'holographic_gpu*.so')):
                spec = _ilu.spec_from_file_location('holographic_gpu_wave', fn)
                if spec and spec.loader:
                    mod = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[assignment]
                    if hasattr(mod, 'wave_ecc_encode') and hasattr(mod, 'wave_ecc_decode'):
                        _hn = mod  # prefer this binding for subsequent calls
                        return mod
    except Exception:
        pass
    raise RuntimeError('Wave ECC binding not available in holographic_gpu')


# ----------------- Wave ECC metrics (production observability) -----------------
class _WaveECCMetrics:
    """Thread-safe accumulator for Wave ECC observability.

    Metrics are updated in verify_and_correct_rs() and exposed via stats().
    Optional JSONL telemetry to `./logs/wave_ecc_metrics.jsonl` is enabled when
    env `HOLO_WAVE_ECC_TELEMETRY=1`.
    """

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        # Counters
        self.total_encode_ops = 0
        self.total_decode_ops = 0
        self.total_bytes_encoded = 0
        self.total_bytes_decoded = 0
        self.total_parity_bytes = 0
        self.errors_detected = 0
        self.decode_success = 0
        self.decode_fail = 0
        # Timing (ms)
        self.last_encode_ms: float | None = None
        self.last_decode_ms: float | None = None
        self.avg_encode_ms: float | None = None
        self.avg_decode_ms: float | None = None
        self._alpha = 0.2  # EMA smoothing
        # Telemetry control
        self._telemetry = os.environ.get("HOLO_WAVE_ECC_TELEMETRY", "0").strip() in ("1", "true", "on")
        self._telemetry_path = Path.cwd() / "logs" / "wave_ecc_metrics.jsonl"
        self._telemetry_started = False

    def _emit(self, record: Dict[str, Any]) -> None:
        if not self._telemetry:
            return
        try:
            p = self._telemetry_path
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:
            pass

    def _proc_sample(self) -> Dict[str, Any]:
        proc = None
        try:
            if _psutil is not None:
                proc = _psutil.Process()
        except Exception:
            proc = None
        if not proc:
            return {"rss_mb": None, "cpu_pct": None}
        try:
            mi = proc.memory_info()
            rss_mb = round(mi.rss / (1024 * 1024), 2)
        except Exception:
            rss_mb = None
        cpu_pct = None
        try:
            cpu_pct = proc.cpu_percent(interval=0.0)
        except Exception:
            pass
        return {"rss_mb": rss_mb, "cpu_pct": cpu_pct}

    def record_encode(self, data_len: int, parity_len: int, ms: float) -> None:
        with self._lock:
            self.total_encode_ops += 1
            self.total_bytes_encoded += int(data_len)
            self.total_parity_bytes += int(parity_len)
            self.last_encode_ms = float(ms)
            if self.avg_encode_ms is None:
                self.avg_encode_ms = float(ms)
            else:
                self.avg_encode_ms = (1 - self._alpha) * float(self.avg_encode_ms) + self._alpha * float(ms)
            rec = {
                "ts": _time.time(),
                "event": "encode",
                "data_bytes": int(data_len),
                "parity_bytes": int(parity_len),
                "latency_ms": float(ms),
            }
            rec.update(self._proc_sample())
        self._emit(rec)

    def record_decode(self, data_len: int, errors: int, success: bool, ms: float) -> None:
        with self._lock:
            self.total_decode_ops += 1
            self.total_bytes_decoded += int(data_len)
            self.errors_detected += int(errors)
            if success:
                self.decode_success += 1
            else:
                self.decode_fail += 1
            self.last_decode_ms = float(ms)
            if self.avg_decode_ms is None:
                self.avg_decode_ms = float(ms)
            else:
                self.avg_decode_ms = (1 - self._alpha) * float(self.avg_decode_ms) + self._alpha * float(ms)
            rec = {
                "ts": _time.time(),
                "event": "decode",
                "data_bytes": int(data_len),
                "errors_detected": int(errors),
                "success": bool(success),
                "latency_ms": float(ms),
            }
            rec.update(self._proc_sample())
        self._emit(rec)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_ops = self.total_encode_ops + self.total_decode_ops
            success_rate = None
            if (self.decode_success + self.decode_fail) > 0:
                success_rate = round(100.0 * self.decode_success / (self.decode_success + self.decode_fail), 2)
            overhead = None
            if self.total_bytes_encoded > 0:
                overhead = round(self.total_parity_bytes / max(1, self.total_bytes_encoded), 3)
            return {
                "total_encode_ops": self.total_encode_ops,
                "total_decode_ops": self.total_decode_ops,
                "total_bytes_encoded": self.total_bytes_encoded,
                "total_bytes_decoded": self.total_bytes_decoded,
                "total_parity_bytes": self.total_parity_bytes,
                "errors_detected": self.errors_detected,
                "decode_success": self.decode_success,
                "decode_fail": self.decode_fail,
                "decode_success_rate_pct": success_rate,
                "avg_encode_ms": self.avg_encode_ms,
                "avg_decode_ms": self.avg_decode_ms,
                "last_encode_ms": self.last_encode_ms,
                "last_decode_ms": self.last_decode_ms,
                "avg_parity_overhead": overhead,
            }


__wave_metrics = _WaveECCMetrics()


def verify_and_correct_rs(payload: bytes, parity: bytes, k: int = 3, r: int = 0) -> bytes:
    """ECC verify/correct helper (Wave ECC backend).

    Backward-compatible name retained for API stability. Parameters map to Wave ECC:
    - k: redundancy_level (number of parity views), default 3
    - r: seed_base (deterministic seed used during encode)

    Returns corrected bytes (same length as input payload). Raises RuntimeError
    if parity re-validation fails after correction.
    """
    if not payload or not parity or k <= 0:
        return payload
    if not _cpp_loaded:
        raise RuntimeError("GPU ECC backend not available; build native extensions")
    # Wave ECC decode/correct via already-imported native module (GPU-only)
    _hgw = _get_wave_binding()
    t0 = _time.perf_counter()
    decoded_bytes, errors_detected = _hgw.wave_ecc_decode(payload, parity, int(k), int(r))  # type: ignore[attr-defined]
    dec_ms = ( _time.perf_counter() - t0 ) * 1000.0
    corrected = bytes(decoded_bytes)
    # Parity re-validation to ensure integrity
    t1 = _time.perf_counter()
    new_parity = _hgw.wave_ecc_encode(corrected, int(k), int(r))  # type: ignore[attr-defined]
    enc_ms = ( _time.perf_counter() - t1 ) * 1000.0
    ok = bytes(new_parity) == bytes(parity)
    __wave_metrics.record_decode(len(payload), int(errors_detected), ok, dec_ms)
    __wave_metrics.record_encode(len(corrected), len(new_parity or b""), enc_ms)
    if not ok:
        raise RuntimeError(
            f"ECC uncorrectable: parity mismatch after correction (errors_detected={int(errors_detected)})"
        )
    return corrected

def calculate_optimal_dimension(file_size: int) -> int:
    """Adaptive FFT dimension selection based on file size."""
    if file_size < 512:
        return 8
    if file_size < 4096:
        return 16
    if file_size < 32768:
        return 32
    if file_size < 262144:
        return 64
    if file_size < 1048576:
        return 128
    return 256


class Memory:
    """Thin wrapper around Python holographic memory for HoloFS."""

    def __init__(self, state_dir: Path, grid_size: int = 32, use_gpu: bool = True) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Persist grid size for GPU helpers
        self.grid_size = int(grid_size)
        # GPU-first selection: prefer GPU backend as the primary engine, CPU as fallback
        self.gpu_backend = None
        self.use_gpu = False
        if use_gpu and _cpp_loaded and hasattr(_hn, 'HolographicGPU'):
            try:
                be = _hn.HolographicGPU()  # type: ignore[name-defined, attr-defined]
                ok = True
                if hasattr(be, 'initialize'):
                    try:
                        ok = bool(be.initialize())  # type: ignore[attr-defined]
                    except Exception:
                        ok = False
                if ok:
                    self.backend = be
                    self.gpu_backend = be
                    self.use_gpu = True
                else:
                    raise RuntimeError('GPU initialize returned False')
            except Exception:
                # GPU unavailable; fall back to CPU engines
                self.backend = None  # type: ignore[assignment]

        if not hasattr(self, 'backend') or self.backend is None:  # type: ignore[attr-defined]
            if _DEV_MODE and _cpu_loaded and _hn_cpu is not None and hasattr(_hn_cpu, 'HolographicMemory'):
                try:
                    self.backend = _hn_cpu.HolographicMemory(int(grid_size))  # type: ignore[name-defined, attr-defined]
                except Exception as exc:
                    raise RuntimeError("CPU backend failed to initialize") from exc
            elif _DEV_MODE and _cpp_loaded and hasattr(_hn, 'HolographicMemory'):
                # Fallback to legacy CPU backend if present under holographic_gpu namespace
                try:
                    self.backend = _hn.HolographicMemory(int(grid_size))  # type: ignore[name-defined, attr-defined]
                except Exception as exc:
                    raise RuntimeError("Legacy CPU backend failed to initialize") from exc
            elif _cpp_loaded and hasattr(_hn, 'HolographicGPU'):
                # Fallback to GPU backend when CPU backend is not available
                try:
                    be = _hn.HolographicGPU()  # type: ignore[name-defined, attr-defined]
                    if hasattr(be, 'initialize'):
                        be.initialize()
                    self.backend = be
                    self.gpu_backend = be
                    self.use_gpu = True
                except Exception as exc:
                    raise RuntimeError("GPU backend fallback failed to initialize") from exc
            else:
                raise RuntimeError("C++ backend not available. Build the extensions (make native)")
        # 3D exact-recall engine is optional; used for byte-perfect storage/recall
        self.backend3d = None
        if _cpp3d_loaded:
            try:
                self.backend3d = _hn3d.HolographicMemory3D(int(grid_size))  # type: ignore[name-defined, attr-defined]
            except Exception:
                self.backend3d = None

        # If CPU backend is primary, still attempt to attach a GPU accelerator if available
        if not self.use_gpu and use_gpu:
            try:
                import holographic_gpu as _hg  # type: ignore
                if hasattr(_hg, 'HolographicGPU'):
                    be2 = _hg.HolographicGPU()  # type: ignore[attr-defined]
                    ok2 = True
                    if hasattr(be2, 'initialize'):
                        try:
                            ok2 = bool(be2.initialize())  # type: ignore[attr-defined]
                        except Exception:
                            ok2 = False
                    if ok2:
                        self.gpu_backend = be2
                        self.use_gpu = True
            except Exception:
                try:
                    import holographic_metal as _hm  # type: ignore
                    if hasattr(_hm, 'MetalHolographicBackend'):
                        be2 = _hm.MetalHolographicBackend()  # type: ignore[attr-defined]
                        ok2 = True
                        if hasattr(be2, 'initialize'):
                            try:
                                ok2 = bool(be2.initialize())  # type: ignore[attr-defined]
                            except Exception:
                                ok2 = False
                        if ok2:
                            self.gpu_backend = be2
                            self.use_gpu = True
                except Exception:
                    self.gpu_backend = None

        # Initialize math-layer state on the active backend, if supported
        try:
            if hasattr(self.backend, "initialize_7layer_decomposition"):
                self.backend.initialize_7layer_decomposition(int(self.grid_size))  # type: ignore[attr-defined]
                if hasattr(self.backend, "update_layer_snrs"):
                    self.backend.update_layer_snrs()  # type: ignore[attr-defined]
        except Exception:
            # Non-fatal: capabilities will reflect initialization status
            pass

    def store_file(self, path: Path, stable_id: Optional[str] = None) -> str:
        """Store file bytes using exact-recall 3D backend and record wave meta.

        Removes legacy base64/disk persistence to keep storage holographic-only.
        """
        data = path.read_bytes()
        from .index import sha256_file as _sha
        digest = _sha(path)
        doc_id = stable_id or digest

        # 1) Store raw bytes in 3D engine for exact recall
        if self.backend3d is not None and hasattr(self.backend3d, "store_bytes"):
            try:
                self.backend3d.store_bytes(data, doc_id)  # type: ignore[attr-defined]
            except Exception:
                # Non-fatal: proceed without 3D exact-recall persistence
                pass
        else:
            # Fallback: legacy persistence (HRR base64 + disk) to preserve recall
            try:
                import json as _json
                if hasattr(self.backend, "store_response_hrr"):
                    import base64 as _b64
                    b64_data = _b64.b64encode(data).decode("ascii")
                    self.backend.store_response_hrr(f"{doc_id}#data", b64_data)  # type: ignore[attr-defined]
                    self.backend.store_response_hrr(
                        f"{doc_id}#manifest",
                        _json.dumps({"doc_id": doc_id, "size": len(data), "filename": Path(path).name, "type": "base64_direct"}),
                    )  # type: ignore[attr-defined]
                resp_dir = self.state_dir / "responses" / doc_id
                resp_dir.mkdir(parents=True, exist_ok=True)
                (resp_dir / "data.bin").write_bytes(data)
                (resp_dir / "meta.json").write_text(
                    _json.dumps({"filename": Path(path).name, "size": len(data)}), encoding="utf-8"
                )
            except Exception:
                pass

        # 2) Store lightweight meta in wave engine for pattern access (no original data)
        try:
            meta = f"filename:{Path(path).name}\nsize:{len(data)}\nsha256:{digest}\n"
            engine_id = self.backend.store(meta)  # type: ignore[attr-defined]
            # Persist doc_id -> engine_id mapping inside HM and on disk
            try:
                mapping = {"doc_id": doc_id, "engine_id": engine_id, "filename": Path(path).name, "size": len(data)}
                if hasattr(self.backend, "store_response_hrr"):
                    self.backend.store_response_hrr(f"{doc_id}#engine_mapping", _json.dumps(mapping))  # type: ignore[attr-defined]
                mpath = self.state_dir / "engine_map.json"
                db = {}
                if mpath.exists():
                    try:
                        db = _json.loads(mpath.read_text(encoding='utf-8'))
                    except Exception:
                        db = {}
                db[str(doc_id)] = mapping
                mpath.write_text(_json.dumps(db, indent=2), encoding='utf-8')
            except Exception:
                pass
        except Exception:
            pass

        # 3) Record adaptive dimension mapping for compact wave vectors
        try:
            dim = calculate_optimal_dimension(len(data))
            self._store_dimension_mapping(doc_id, dim)
        except Exception:
            pass

        # No base64 HRR or disk copy when 3D backend is available — holographic-only
        return doc_id

    # ----------------- GPU Container (HGMC2/HGMC3) helpers -----------------
    def _containers_dir(self) -> Path:
        p = self.state_dir / "hlog" / "containers"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _container_map_path(self) -> Path:
        return self.state_dir / "hlog" / "container_map.json"

    def _register_container(self, doc_id: str, filename: str, dim: int) -> None:
        mpath = self._container_map_path()
        db = {}
        try:
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
        except Exception:
            db = {}
        # Maintain backward-compatible minimal record; layered metadata may be attached by caller
        entry = db.get(str(doc_id), {})
        entry.update({"filename": filename, "dimension": int(dim), "container": f"{doc_id}.hgc"})
        db[str(doc_id)] = entry
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(_json.dumps(db, indent=2), encoding='utf-8')
        os.replace(tmp, mpath)

    def _gpu_available(self) -> bool:
        try:
            import holographic_gpu as _hg  # type: ignore
            g = _hg.HolographicGPU()  # type: ignore[attr-defined]
            plats = getattr(_hg, 'available_platforms', lambda: [])()  # type: ignore[attr-defined]
            if not plats:
                return False
            if hasattr(g, 'initialize'):
                return bool(g.initialize())  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    def store_bytes(self, data: bytes, doc_id: str, filename: str) -> str:
        """GPU-only container store (HGMC2) with ECC parity.

        Requires `holographic_gpu` to be available. Creates `.holofs/hlog/containers/{doc_id}.hgc`.
        """
        # Honor GPU-only override
        gpu_only = os.environ.get("HLOG_GPU_ONLY", "0") == "1"
        if not (gpu_only or self._gpu_available()):
            raise RuntimeError("GPU backend not available for container store")
        import hashlib as _hl
        import numpy as _np
        if not _cpp_loaded:
            raise RuntimeError("GPU backend not available")
        if not data:
            out = self._containers_dir() / f"{doc_id}.hgc"
            out.write_bytes(b"HGMC2\x00" + (0).to_bytes(4,'little')*3)
            self._register_container(doc_id, filename, 0)
            self._store_dimension_mapping(doc_id, 0)
            return doc_id
        # Seed base from content hash for determinism
        seed_base = int(_hl.sha256(data).hexdigest()[:8], 16)
        # Encode and superpose on GPU
        g = _hn.HolographicGPU()  # type: ignore[attr-defined]
        if hasattr(g, 'initialize'):
            _ = g.initialize()  # type: ignore[attr-defined]
        # Initialize 7-layer decomposition to match content dimension
        try:
            if hasattr(g, 'initialize_7layer_decomposition'):
                g.initialize_7layer_decomposition(len(data))  # type: ignore[attr-defined]
            # Phase 4 polish: enforce capacity theorem and refresh SNRs before routing
            if hasattr(g, 'update_layer_snrs'):
                g.update_layer_snrs()  # type: ignore[attr-defined]
            if hasattr(g, 'enforce_capacity_theorem'):
                try:
                    _ = g.enforce_capacity_theorem()  # type: ignore[attr-defined]
                except Exception:
                    pass
            if hasattr(g, 'update_layer_snrs'):
                g.update_layer_snrs()  # type: ignore[attr-defined]
        except Exception:
            pass

        # --- Phase 4: 7-layer routing for bytes ---
        # Decide per-chunk layer routing and per-layer alpha scaling
        def _route_layers_for_file(fname: str, raw: bytes, nchunks: int) -> tuple[list[int], list[float]]:
            # Simple deterministic heuristics (aligned with api/router_layers.py)
            name = str(fname).lower()
            ext = name.rsplit('.', 1)[-1] if '.' in name else ''
            scores = {
                'identity': 0.0,
                'knowledge': 0.6,
                'experience': 0.0,
                'preference': 0.0,
                'context': 0.2,
                'wisdom': 0.0,
                'vault': 0.0,
            }
            # Vault triggers
            for kw in ('secret', 'key', 'password', 'token', 'cert'):
                if kw in name:
                    scores = {k: 0.0 for k in scores}; scores['vault'] = 1.0; break
            # Extension-based hints
            if ext in ('pdf','txt','md','doc','docx','rtf','tex'):
                scores['knowledge'] += 0.4
            if ext in ('log','csv','tsv'):
                scores['experience'] += 0.4
            if ext in ('ini','cfg','conf','yaml','yml','toml','json'):
                scores['preference'] += 0.6
            if ext in ('png','jpg','jpeg','gif','bmp','tiff'):
                scores['context'] += 0.3
            # Content heuristics (cheap)
            txt = ''
            try:
                txt = raw[:8192].decode('utf-8', errors='ignore')
            except Exception:
                txt = ''
            import re as _re
            if _re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", txt or ''):
                scores['identity'] += 0.8
            if _re.search(r"\b\w+ed\b", txt or '') or _re.search(r"\b20\d{2}\b", txt or ''):
                scores['experience'] += 0.6
            if 'verified' in (txt or '').lower() or 'checksum' in (txt or '').lower():
                scores['wisdom'] += 0.7
            # Normalize and pick top-2
            pos = [(k,v) for k,v in scores.items() if v>0.0]
            if not pos:
                pos = [('knowledge', 1.0)]
            pos.sort(key=lambda kv: kv[1], reverse=True)
            top = pos[:2]
            s = sum(w for _,w in top)
            if s <= 0:
                top = [('knowledge', 1.0)]; s = 1.0
            norm = [(k, w/s) for k,w in top]
            name_to_idx = {'identity':0,'knowledge':1,'experience':2,'preference':3,'context':4,'wisdom':5,'vault':6}
            # Allocate chunks proportionally
            alloc = [0]*7
            rem = nchunks
            for k,w in norm:
                c = int(round(w * nchunks))
                alloc[name_to_idx[k]] += c; rem -= c
            # Fix rounding by assigning remaining to highest weight
            if rem != 0:
                top_idx = name_to_idx[norm[0][0]]
                alloc[top_idx] += rem
            # Build chunk_layers round-robin among selected layers
            selected = [name_to_idx[k] for k,_ in norm]
            chunk_layers = []
            ptrs = {i:0 for i in selected}
            # Create lists per layer
            layers_flat: list[int] = []
            for i in range(7):
                layers_flat.extend([i]*alloc[i])
            # Deterministic shuffling by seed
            import random as _rnd
            r = _rnd.Random(seed_base)
            r.shuffle(layers_flat)
            chunk_layers = layers_flat[:nchunks]
            # Alpha scaling from backend SNRs
            alphas = [1.0]*7
            try:
                if hasattr(g, 'get_layer_stats'):
                    st = g.get_layer_stats()  # type: ignore[attr-defined]
                    amin = float(os.environ.get('LAYER_ALPHA_MIN','0.5'))
                    amax = float(os.environ.get('LAYER_ALPHA_MAX','2.0'))
                    for i in range(7):
                        li = st.get(str(i), {})
                        cur = float(li.get('current_snr', 1.0) or 1.0)
                        tgt = float(li.get('target_snr', 1.0) or 1.0)
                        a = tgt / max(1e-6, cur)
                        if not (a == a):  # NaN guard
                            a = 1.0
                        a = max(amin, min(amax, a))
                        alphas[i] = a
            except Exception:
                pass
            return chunk_layers, alphas

        # Build chunk routing (we don't know chunk count until GPU splits, approximate by len/4096)
        approx_chunks = max(1, (len(data) + 4095)//4096)
        chunk_layers, alpha_scales = _route_layers_for_file(filename, data, approx_chunks)
        # Optional: enforce capacity before encode using the final pre-computed routing plan
        try:
            if hasattr(g, 'enforce_capacity_theorem'):
                _ = g.enforce_capacity_theorem()  # type: ignore[attr-defined]
            if hasattr(g, 'update_layer_snrs'):
                g.update_layer_snrs()  # type: ignore[attr-defined]
        except Exception:
            pass
        # Execute layered encode if available; else fallback
        try:
            if hasattr(g, 'encode_superpose_bytes_layered'):
                psi, dim, seeds, sizes = g.encode_superpose_bytes_layered(data, 4096, seed_base, list(map(int, chunk_layers)), list(map(float, alpha_scales)))  # type: ignore[attr-defined]
            else:
                psi, dim, seeds, sizes = g.encode_superpose_bytes(data, 4096, seed_base)  # type: ignore[attr-defined]
        except Exception:
            psi, dim, seeds, sizes = g.encode_superpose_bytes(data, 4096, seed_base)  # type: ignore[attr-defined]
        psi_arr = _np.asarray(psi, dtype=_np.float32)
        # ECC parity per original chunk (Wave ECC)
        # Map header fields as follows for compatibility:
        #   ecc_scheme = 2 (Wave ECC)
        #   ecc_k      = redundancy_level
        #   ecc_r      = seed_base (same as encode_superpose_bytes)
        ecc_scheme = 2
        redundancy_level = int(os.environ.get("WAVE_ECC_REDUNDANCY", "3"))
        ecc_k = redundancy_level
        ecc_r = seed_base
        parities: List[bytes] = []
        off = 0
        for sz in sizes:
            chunk = data[off:off+sz]
            _hgw = _get_wave_binding()
            p = _hgw.wave_ecc_encode(chunk, int(ecc_k), int(ecc_r))  # type: ignore[attr-defined]
            parities.append(bytes(p))
            off += sz
        # Write HGMC2 container
        out = self._containers_dir() / f"{doc_id}.hgc"
        with open(out, 'wb') as f:
            f.write(b"HGMC2\x00")
            f.write(int(dim).to_bytes(4, 'little'))
            f.write(int(len(data)).to_bytes(4, 'little'))
            f.write(int(len(sizes)).to_bytes(4, 'little'))
            for sz in sizes:
                f.write(int(int(sz)).to_bytes(4, 'little'))
            for sd in seeds:
                f.write(int(int(sd)).to_bytes(4, 'little'))
            f.write(int(ecc_scheme).to_bytes(4, 'little'))
            f.write(int(ecc_k).to_bytes(4, 'little'))
            f.write(int(ecc_r).to_bytes(4, 'little'))
            for pb in parities:
                f.write(int(len(pb)).to_bytes(4, 'little'))
                if pb:
                    f.write(pb)
            f.write(psi_arr.tobytes(order='C'))
        # Register container and attach layered metadata to map for telemetry
        self._register_container(doc_id, filename, int(dim))
        try:
            # Attach layered metadata into container map (non-invasive)
            mpath = self._container_map_path()
            db = {}
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
            ent = db.get(str(doc_id), {})
            ent['layer_alpha_scales'] = list(map(float, alpha_scales)) if 'alpha_scales' in locals() else []
            ent['chunk_layers'] = list(map(int, chunk_layers)) if 'chunk_layers' in locals() else []
            ent['ecc'] = {"scheme": int(ecc_scheme), "k": int(ecc_k), "r": int(ecc_r)}
            db[str(doc_id)] = ent
            tmp = mpath.with_suffix('.tmp')
            tmp.write_text(_json.dumps(db, indent=2), encoding='utf-8')
            os.replace(tmp, mpath)
        except Exception:
            pass
        self._store_dimension_mapping(doc_id, int(dim))
        return doc_id

    def retrieve_to(self, doc_id: str, out_path: Path) -> Path:
        # Try holographic-only bytes reconstruction from stored chunks
        try:
            data = self.retrieve_bytes(doc_id)
        except Exception:  # pylint: disable=broad-except
            # Fallback A: decode .hwp from index using holographic_gpu decoder (H4K8/HWP4V001)
            try:
                ent = next((e for e in self.index.all() if e.doc_id == doc_id), None)  # type: ignore[attr-defined]
            except Exception:
                ent = None
            if ent is not None:
                try:
                    p = Path(ent.path)
                    if p.suffix.lower() == ".hwp" and p.exists():
                        import holographic_gpu as _hg  # type: ignore
                        dec = _hg.HolographicGPU() if hasattr(_hg, 'HolographicGPU') else None  # type: ignore[attr-defined]
                        if dec is not None and hasattr(dec, 'retrieve_bytes'):
                            try:
                                # initialize is best-effort; decoder runs CPU-side
                                if hasattr(dec, 'initialize'):
                                    try:
                                        dec.initialize('metal')  # type: ignore[attr-defined]
                                    except Exception:
                                        pass
                                raw = dec.retrieve_bytes(str(p))  # type: ignore[attr-defined]
                                if isinstance(raw, str):
                                    data = raw.encode('latin-1', errors='ignore')
                                else:
                                    data = bytes(raw)
                            except Exception:
                                data = None  # type: ignore[assignment]
                        else:
                            data = None  # type: ignore[assignment]
                    else:
                        data = None  # type: ignore[assignment]
                except Exception:
                    data = None  # type: ignore[assignment]
            else:
                data = None  # type: ignore[assignment]

            if data is None:
                # Fallback B: if engine exposes direct retrieve (legacy CPU backend), try it
                if hasattr(self.backend, "retrieve"):
                    data = self.backend.retrieve(doc_id)  # type: ignore[attr-defined]
                else:
                    raise RuntimeError("Holographic retrieve not available for this document") from None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        if _cpp_loaded and hasattr(self.backend, "query"):
            try:
                return [(r[0], float(r[1]), r[2]) for r in self.backend.query(query, int(k))]  # type: ignore[attr-defined]
            except Exception:  # pylint: disable=broad-except
                pass
        return []

    def stats(self) -> Dict:
        base: Dict[str, Any] = {}
        if _cpp_loaded and hasattr(self.backend, "get_stats"):
            try:
                base = dict(self.backend.get_stats())  # type: ignore[attr-defined]
            except Exception:  # pylint: disable=broad-except
                base = {}
        # Include 7-layer telemetry if available
        try:
            if _cpp_loaded and hasattr(self.backend, "get_layer_stats"):
                layers = self.backend.get_layer_stats()  # type: ignore[attr-defined]
                base["layers"] = dict(layers) if isinstance(layers, dict) else layers
        except Exception:
            pass
        # Attach Wave ECC metrics and GPU availability info
        try:
            plats = []
            try:
                plats = getattr(_hn, 'available_platforms', lambda: [])() if _cpp_loaded else []  # type: ignore[attr-defined]
            except Exception:
                plats = []
            base.update({
                "gpu": {
                    "available": bool(self.use_gpu),
                    "platforms": plats,
                },
                "wave_ecc_metrics": __wave_metrics.snapshot(),
            })
        except Exception:
            pass
        # Aggregate per-layer chunk counts from container map (observability)
        try:
            mpath = self._container_map_path()
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
                counts = {str(i): 0 for i in range(7)}
                for ent in db.values():
                    cl = ent.get('chunk_layers') or []
                    for v in cl:
                        k = str(int(v) % 7)
                        counts[k] = counts.get(k, 0) + 1
                base.setdefault('layers', {})
                if isinstance(base['layers'], dict):
                    base['layers']['chunk_counts'] = counts
                else:
                    base['layers'] = {"chunk_counts": counts}
        except Exception:
            pass
        return base

    # ----------------- GPU helpers (optional) -----------------
    def store_batch_gpu(self, batch_data: List[List[float]]) -> List[List[float]]:
        """Store/encode a batch using GPU acceleration if available.

        Returns encoded holographic patterns (list of vectors). If GPU is
        unavailable, returns an empty list.
        """
        if not self.use_gpu or self.gpu_backend is None:
            return []
        try:
            # Prefer new API: batch_encode(batch, pattern_dim)
            if hasattr(self.gpu_backend, "batch_encode"):
                return self.gpu_backend.batch_encode(batch_data, int(self.grid_size))  # type: ignore[attr-defined]
            # Legacy module-level function via wrapper
            if hasattr(self.gpu_backend, "gpu_batch_store"):
                return self.gpu_backend.gpu_batch_store(batch_data, int(self.grid_size))  # type: ignore[attr-defined]
            # Optional NumPy pathway for new backend
            if hasattr(self.gpu_backend, "batch_encode_numpy"):
                import numpy as _np
                arr = _np.array(batch_data, dtype=_np.float32, order="C")
                out = self.gpu_backend.batch_encode_numpy(arr, int(self.grid_size))  # type: ignore[attr-defined]
                return [list(map(float, row)) for row in out]
        except Exception:
            return []
        return []

    def get_performance_metrics(self) -> dict:
        """Return last known GPU/CPU performance metrics (if available)."""
        metrics: dict = {"cpu": {}}
        if self.use_gpu and self.gpu_backend is not None:
            try:
                m = None
                if hasattr(self.gpu_backend, "get_last_metrics"):
                    m = self.gpu_backend.get_last_metrics()  # type: ignore[attr-defined]
                elif hasattr(self.gpu_backend, "metrics"):
                    m = self.gpu_backend.metrics()  # type: ignore[attr-defined]
                if m is not None:
                    # Support both Metal legacy and new cross-platform structs
                    ops = getattr(m, "operations_per_second", 0.0)
                    bw = getattr(m, "memory_bandwidth_gb_s", 0.0)
                    # Field name differs in legacy vs new; prefer device_ms or batch_time_ms
                    bt = getattr(m, "batch_encode_time_ms", None)
                    if bt is None:
                        bt = getattr(m, "device_ms", None)
                    if bt is None:
                        bt = getattr(m, "batch_time_ms", 0.0)
                    metrics["gpu"] = {
                        "operations_per_second": float(ops or 0.0),
                        "memory_bandwidth_gb_s": float(bw or 0.0),
                        "batch_encode_time_ms": float(bt or 0.0),
                    }
            except Exception:
                pass
        return metrics

    def retrieve_bytes(self, doc_id: str) -> bytes:
        """Reconstruct file bytes.

        Priority:
          1) GPU container (HGMC2/HGMC3) if present or HLOG_GPU_ONLY=1
          2) 3D engine exact recall (preferred)
          3) Legacy HRR chunk/base64 (if present)
        """
        # Try GPU holographic containers first when requested/available
        try:
            cpath = self.state_dir / "hlog" / "containers" / f"{doc_id}.hgc"
            gpu_only = os.environ.get("HLOG_GPU_ONLY", "0") == "1"
            if cpath.exists() or gpu_only:
                import numpy as _np
                import holographic_gpu as _hg  # type: ignore
                with open(cpath, 'rb') as f:
                    head = f.read(6)
                    magic = head[:6]
                    ecc_scheme = 0; ecc_k = 0; ecc_r = 0
                    parities: List[bytes] = []
                    if magic.startswith(b"HGMC3"):
                        raise RuntimeError("HGMC3 not implemented in this build")
                    elif magic.startswith(b"HGMC2"):
                        dim = int.from_bytes(f.read(4), 'little')
                        orig = int.from_bytes(f.read(4), 'little')
                        n = int.from_bytes(f.read(4), 'little')
                        sizes = [int.from_bytes(f.read(4), 'little') for _ in range(n)]
                        seeds = [int.from_bytes(f.read(4), 'little') for _ in range(n)]
                        ecc_scheme = int.from_bytes(f.read(4), 'little')
                        ecc_k = int.from_bytes(f.read(4), 'little')
                        ecc_r = int.from_bytes(f.read(4), 'little')
                        for _ in range(n):
                            plen = int.from_bytes(f.read(4), 'little')
                            parities.append(f.read(plen) if plen > 0 else b"")
                        psi = _np.frombuffer(f.read(dim * 4), dtype=_np.float32)
                    else:
                        raise RuntimeError("Unsupported holographic container magic")
                # Decode via GPU correlation
                g = _hn.HolographicGPU()  # type: ignore[attr-defined]
                if hasattr(g, 'initialize'):
                    _ = g.initialize()  # type: ignore[attr-defined]
                out = g.decode_superposed_bytes(psi.astype(_np.float32).tolist(), int(dim), list(map(int, seeds)), list(map(int, sizes)))  # type: ignore[attr-defined]
                if not isinstance(out, (bytes, bytearray)):
                    raise RuntimeError("GPU decode_superposed_bytes returned invalid type")
                data_bytes = bytes(out)
                # ECC verify/correct if present (Wave ECC only)
                if ecc_scheme == 2 and ecc_k > 0 and ecc_r >= 0 and parities:
                    corrected: List[bytes] = []
                    off = 0
                    for idx, sz in enumerate(sizes):
                        chunk = data_bytes[off:off+sz]
                        par = parities[idx] if idx < len(parities) else b""
                        if par:
                            chunk = verify_and_correct_rs(chunk, par, int(ecc_k), int(ecc_r))
                        corrected.append(chunk)
                        off += sz
                    data_bytes = b"".join(corrected)
                return data_bytes
        except Exception:
            if os.environ.get("HLOG_GPU_ONLY", "0") == "1":
                raise
        # Prefer 3D engine exact recall
        if self.backend3d is not None and hasattr(self.backend3d, "retrieve_bytes"):
            try:
                return bytes(self.backend3d.retrieve_bytes(doc_id))  # type: ignore[attr-defined]
            except Exception:
                pass

        # Legacy disk-backed retrieval (if present)
        resp_dir = self.state_dir / "responses" / doc_id
        data_path = resp_dir / "data.bin"
        if data_path.exists():
            return data_path.read_bytes()

        # Legacy HRR chunk/base64 retrieval
        if not (_cpp_loaded and hasattr(self.backend, "retrieve_response_hrr")):
            raise RuntimeError("No holographic retrieval path available for doc_id")
        import json as _json
        import base64 as _b64
        man_txt = self.backend.retrieve_response_hrr(f"{doc_id}#manifest")  # type: ignore[attr-defined]
        if not isinstance(man_txt, str) or not man_txt:
            raise KeyError("Manifest not found for doc_id")
        man = _json.loads(man_txt)
        storage_type = man.get("type", "chunked")
        if storage_type == "base64_direct":
            data_txt = self.backend.retrieve_response_hrr(f"{doc_id}#data")  # type: ignore[attr-defined]
            if not isinstance(data_txt, str):
                raise KeyError("Data not found for doc_id")
            return _b64.b64decode(data_txt.encode("ascii"))
        # Legacy chunked storage
        chunks = int(man.get("chunks", 0))
        buf = bytearray()
        for i in range(chunks):
            part_txt = self.backend.retrieve_response_hrr(f"{doc_id}#chunk:{i}")  # type: ignore[attr-defined]
            if not isinstance(part_txt, str) or not part_txt:
                raise KeyError(f"Missing chunk {i}")
            buf.extend(_b64.b64decode(part_txt.encode("ascii"), validate=False))
        size = int(man.get("size", len(buf)))
        return bytes(buf[:size])

    # ----------------- Real wave access helpers -----------------
    def get_engine_id(self, doc_id: str) -> str:
        import json as _json
        # On-disk persistent map first
        try:
            mpath = self.state_dir / "engine_map.json"
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
                m = db.get(str(doc_id))
                if m and m.get('engine_id'):
                    return str(m['engine_id'])
        except Exception:
            pass
        if hasattr(self.backend, "retrieve_response_hrr"):
            try:
                m = self.backend.retrieve_response_hrr(f"{doc_id}#engine_mapping")  # type: ignore[attr-defined]
                if isinstance(m, str) and m:
                    d = _json.loads(m)
                    eng = d.get("engine_id")
                    if eng:
                        return str(eng)
            except Exception:
                pass
        raise RuntimeError(f"No engine mapping found for doc_id: {doc_id}")

    def _dimension_map_path(self) -> Path:
        return self.state_dir / "dimension_map.json"

    def _store_dimension_mapping(self, doc_id: str, dimension: int, memory_id: Optional[str] = None) -> None:
        """Persist dimension used for this doc_id for adaptive wave extraction."""
        mpath = self._dimension_map_path()
        db: Dict[str, Any] = {}
        try:
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
        except Exception:
            db = {}
        db[str(doc_id)] = {
            "dimension": int(dimension),
            "memory_id": str(memory_id) if memory_id else "",
            "timestamp": _time.time(),
        }
        mpath.write_text(_json.dumps(db, indent=2), encoding='utf-8')

    def set_dimension_mapping(self, doc_id: str, dimension: int) -> None:
        self._store_dimension_mapping(doc_id, int(dimension))

    def _get_mapped_dimension(self, doc_id: str, fallback_size: Optional[int] = None) -> int:
        try:
            mpath = self._dimension_map_path()
            if mpath.exists():
                db = _json.loads(mpath.read_text(encoding='utf-8'))
                m = db.get(str(doc_id))
                if m and int(m.get("dimension", 0)) > 0:
                    return int(m["dimension"])
        except Exception:
            pass
        # Fallback: derive from size if provided, else default to 64
        if fallback_size is not None:
            return calculate_optimal_dimension(int(fallback_size))
        return 64

    def get_real_wave_data(self, doc_id: str) -> Dict[str, Any]:
        """Return adaptive holographic wave vector (amplitude+phase) for doc_id.

        Computes an FFT-based compact vector with a dimension chosen per-file
        (from mapping created at store time). Falls back gracefully.
        """
        import numpy as _np
        # Retrieve bytes first to support engine-agnostic wave generation
        data = self.retrieve_bytes(doc_id)
        size = len(data)
        dim = self._get_mapped_dimension(doc_id, fallback_size=size)
        if dim <= 0:
            dim = 64
        # Build normalized real signal from bytes
        x = _np.frombuffer(data, dtype=_np.uint8).astype(_np.float32)
        if x.size == 0:
            return {"amplitudes": [], "phases": [], "dimension": 0, "source": "adaptive_holographic_engine", "doc_id": doc_id}
        x = x / 255.0
        N = int(dim)
        if x.size == N:
            s = x
        elif x.size < N:
            s = _np.zeros(N, dtype=_np.float32)
            s[:x.size] = x
        else:
            # Resample to N via linear interpolation for better coverage
            idx = _np.linspace(0, x.size - 1, N, dtype=_np.float64)
            s = _np.interp(idx, _np.arange(x.size, dtype=_np.float64), x).astype(_np.float32)
        # FFT to spectral vector
        v = _np.fft.fft(s)
        amps = _np.abs(v).astype(_np.float64).tolist()
        phases = _np.angle(v).astype(_np.float64).tolist()
        return {
            "amplitudes": amps,
            "phases": phases,
            "dimension": int(N),
            "source": "adaptive_holographic_engine",
            "doc_id": doc_id,
        }

    def get_wave_data_from_bytes(self, raw: bytes, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute holographic wave (amp/phase) directly from raw bytes.

        This avoids any dependency on retrieval backends during the store path
        and ensures we always produce spectral data for .hwp writing.
        """
        import numpy as _np
        data = bytes(raw or b"")
        size = len(data)
        # Choose dimension from mapping if available; else derive from size
        if doc_id:
            dim = self._get_mapped_dimension(doc_id, fallback_size=size)
        else:
            dim = calculate_optimal_dimension(size)
        if dim <= 0:
            dim = 64
        # Normalize bytes -> float32 in [0,1]
        x = _np.frombuffer(data, dtype=_np.uint8).astype(_np.float32)
        if x.size == 0:
            return {"amplitudes": [], "phases": [], "dimension": 0, "source": "adaptive_holographic_engine", "doc_id": doc_id or ""}
        x = x / 255.0
        N = int(dim)
        if x.size == N:
            s = x
        elif x.size < N:
            s = _np.zeros(N, dtype=_np.float32)
            s[:x.size] = x
        else:
            idx = _np.linspace(0, x.size - 1, N, dtype=_np.float64)
            s = _np.interp(idx, _np.arange(x.size, dtype=_np.float64), x).astype(_np.float32)
        v = _np.fft.fft(s)
        amps = _np.abs(v).astype(_np.float64).tolist()
        phases = _np.angle(v).astype(_np.float64).tolist()
        # Persist dimension mapping if doc_id provided
        if doc_id:
            try:
                self._store_dimension_mapping(doc_id, N)
            except Exception:
                pass
        return {
            "amplitudes": amps,
            "phases": phases,
            "dimension": int(N),
            "source": "adaptive_holographic_engine",
            "doc_id": doc_id or "",
        }

    def get_collective_interference(self, doc_ids: List[str]) -> Dict[str, Any]:
        try:
            import numpy as _np
            if not hasattr(self.backend, "get_collective_vector"):
                raise RuntimeError("Engine does not expose get_collective_vector")
            eng_ids = [self.get_engine_id(d) for d in doc_ids]
            arr = self.backend.get_collective_vector(eng_ids)  # numpy complex array expected
            amps = _np.abs(arr).tolist()
            phases = _np.angle(arr).tolist()
            return {
                "amplitudes": amps,
                "phases": phases,
                "dimension": int(len(amps)),
                "source": "collective_interference",
                "doc_ids": list(doc_ids),
                "engine_ids": eng_ids,
            }
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to get collective interference: {e}")


class HoloFS:
    """High-level HoloFS API.

    - Manages an index mapping original paths to holographic doc_ids
    - Wraps the holographic memory backend for store/search/recall
    """

    def __init__(self, root: Path, grid_size: int = 32, state_dir: Optional[Path] = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.state_dir = Path(state_dir or (self.root / ".holofs")).resolve()
        self.grid_size = int(grid_size)
        self.mem = Memory(self.state_dir, grid_size=self.grid_size)
        # Persistent on-disk index for durability across processes
        self.index = Index(self.state_dir)

    def store(self, path: Path, force: bool = False) -> str:
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        # Skip if identical already indexed
        ent = self.index.lookup_by_path(path)
        digest = sha256_file(path)
        if ent and not force and ent.doc_id == digest and ent.size == path.stat().st_size:
            return ent.doc_id
        # Use stable content-hash for doc_id and chunk keys
        doc_id = digest
        self.mem.store_file(path, stable_id=doc_id)
        self.index.add_or_update(path, doc_id=doc_id, size=path.stat().st_size)
        return doc_id

    def store_data(self, data: bytes, filename: str, force: bool = False) -> str:
        """Store raw data into holographic containers (GPU-only heavy path)."""
        import hashlib
        doc_id = hashlib.sha256(data).hexdigest()
        # Skip if exists
        if not force:
            existing = [e for e in self.index.all() if e.doc_id == doc_id]
            if existing:
                return doc_id
        # GPU container store
        self.mem.store_bytes(data, doc_id, filename)
        # Index virtual path
        virtual_path = f"holographic://{filename}#{doc_id[:8]}"
        self.index.add_or_update(Path(virtual_path), doc_id=doc_id, size=len(data))
        return doc_id

    def recall(self, query_or_doc: str, out: Optional[Path] = None, original: bool = False) -> Path:
        # Resolve doc id by name if not obvious
        doc_id = None
        # Exact doc id match (supports non-SHA formats like 'mem_123')
        if any(e.doc_id == query_or_doc for e in self.index.all()):
            doc_id = query_or_doc
        elif len(query_or_doc) == 64 and all(c in "0123456789abcdef" for c in query_or_doc):
            doc_id = query_or_doc
        else:
            cands = self.index.lookup_by_name(query_or_doc)
            if not cands:
                raise FileNotFoundError(f"No record matching '{query_or_doc}' in index")
            # choose first match
            ent = cands[0]
            doc_id = ent.doc_id
            if original:
                out = Path(ent.path)
        if out is None:
            out = Path.cwd() / f"{doc_id}.bin"
        # Prefer engine retrieval, else fallback to original file copy
        try:
            return self.mem.retrieve_to(doc_id, out)
        except Exception:  # pylint: disable=broad-except
            # Fallback: copy original file if available
            ent = next((e for e in self.index.all() if e.doc_id == doc_id), None)
            if not ent:
                raise
            src = Path(ent.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read_bytes())
            return out

    def search(self, query: str, k: int = 5):
        return self.mem.search(query, k)

    def stats(self) -> Dict:
        s = self.mem.stats()
        idx = self.index.stats()
        # Sum all bytes under .holofs/hlog for honest accounting
        holo_root = self.state_dir / "hlog"
        holo_bytes = 0
        try:
            if holo_root.exists():
                for p in holo_root.rglob("*"):
                    if p.is_file():
                        try:
                            holo_bytes += p.stat().st_size
                        except OSError:
                            pass
        except Exception:
            pass
        # Holo bytes reflect only .holofs/hlog usage for strict accounting
        s.update(idx)
        # Ensure a dimension field is present for basic status checks
        if "dimension" not in s:
            try:
                s["dimension"] = int(self.grid_size)
            except Exception:
                s["dimension"] = None
        s["holo_bytes"] = int(holo_bytes)
        s["compression_x"] = round((idx.get("original_total_bytes", 0) / holo_bytes), 2) if holo_bytes else None
        return s

    def search_index(self, query: str):
        entries = self.index.lookup_by_name(query)
        return [(e.doc_id, e.path, e.size, getattr(e, 'mtime', 0.0)) for e in entries]


def mount(path: str | Path, grid_size: int = 32, state_dir: Optional[str | Path] = None) -> HoloFS:
    return HoloFS(root=Path(path), grid_size=grid_size, state_dir=Path(state_dir) if state_dir else None)

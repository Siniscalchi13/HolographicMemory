from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json as _json
import time as _time

# Prefer C++ backend (prefer local build over site-packages)
_cpp_loaded = False
_cpp3d_loaded = False
_cpu_loaded = False
_hn_cpu = None  # CPU/HRR engine module (holographic_cpp)
try:
    _pkg_root = Path(__file__).resolve().parents[1]
    _cpp_dir = _pkg_root / "native" / "holographic" / "build"
    if _cpp_dir.exists():
        p = str(_cpp_dir)
        if p not in sys.path:
            sys.path.insert(0, p)
    import holographic_gpu as _hn  # type: ignore
    _cpp_loaded = True
except Exception:
    try:
        import holographic_gpu as _hn  # type: ignore
        _cpp_loaded = True
    except Exception:
        _cpp_loaded = False

# Try to load CPU/HRR engine independently (prefer from build path)
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
            if _cpu_loaded and _hn_cpu is not None and hasattr(_hn_cpu, 'HolographicMemory'):
                try:
                    self.backend = _hn_cpu.HolographicMemory(int(grid_size))  # type: ignore[name-defined, attr-defined]
                except Exception as exc:
                    raise RuntimeError("CPU backend failed to initialize") from exc
            elif _cpp_loaded and hasattr(_hn, 'HolographicMemory'):
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
            except Exception as exc:
                raise RuntimeError(f"3D backend failed to store bytes: {exc}")
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
        if _cpp_loaded and hasattr(self.backend, "get_stats"):
            try:
                return dict(self.backend.get_stats())  # type: ignore[attr-defined]
            except Exception:  # pylint: disable=broad-except
                pass
        return {}

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
        """Reconstruct file bytes using available mechanisms.

        Priority:
          1) 3D engine exact recall (preferred)
          2) Legacy HRR chunk/base64 (if present)
          3) Raise if unavailable
        """
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
        # On-disk holographic state size
        wave = self.state_dir / "wave_spatial.npy"
        holo_bytes = wave.stat().st_size if wave.exists() else 0
        meta = self.state_dir / "metadata.json"
        holo_bytes += meta.stat().st_size if meta.exists() else 0
        holo_bytes += (self.state_dir / "index.json").stat().st_size if (self.state_dir / "index.json").exists() else 0
        s.update(idx)
        s["holo_bytes"] = int(holo_bytes)
        s["compression_x"] = round((idx.get("original_total_bytes", 0) / holo_bytes), 2) if holo_bytes else None
        return s

    def search_index(self, query: str):
        entries = self.index.lookup_by_name(query)
        return [(e.doc_id, e.path, e.size, getattr(e, 'mtime', 0.0)) for e in entries]


def mount(path: str | Path, grid_size: int = 32, state_dir: Optional[str | Path] = None) -> HoloFS:
    return HoloFS(root=Path(path), grid_size=grid_size, state_dir=Path(state_dir) if state_dir else None)

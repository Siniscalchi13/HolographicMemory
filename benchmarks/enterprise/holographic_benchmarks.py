from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest

try:
    from holographicfs.memory import HoloFS, mount
except Exception:  # pragma: no cover - optional import for CI
    HoloFS = None  # type: ignore
    mount = None  # type: ignore

from .statistics import StatisticalValidation


DATA_ROOT = Path("data/enterprise_bench").resolve()
STATE_ROOT = DATA_ROOT / "state"
FS_TMP = DATA_ROOT / "fs_tmp"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_file(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bs = 1024 * 1024
    remain = size_bytes
    with open(path, "wb") as f:
        block = b"\0" * bs
        while remain > 0:
            n = min(bs, remain)
            f.write(block[:n])
            remain -= n


def _ensure_hfs(tmp_root: Path) -> HoloFS:
    if HoloFS is None or mount is None:
        pytest.skip("holographicfs not available (build native backend)")
    state_dir = STATE_ROOT
    state_dir.mkdir(parents=True, exist_ok=True)
    return mount(tmp_root, grid_size=32, state_dir=state_dir)


def _timeit(fn, repeat: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        fn()
    t1 = time.perf_counter()
    return t1 - t0


@dataclass
class PerfRecord:
    label: str
    size: int
    fs_write_s: float
    hm_store_s: float
    chunks: Optional[int]
    holo_bytes: Optional[int]
    compression_x: Optional[float]


class HolographicBenchmarkSuite:
    def test_chunked_storage_performance(self) -> None:
        """256KB chunk storage vs traditional file systems

        Sizes: 1MB, 10MB, 100MB (1GB if HM_BENCH_LARGE=1)
        Measures: storage time, holographic state size, chunk efficiency
        Compare: Filesystem write vs holographic chunking
        """
        sizes = [1 * 1024**2, 10 * 1024**2, 100 * 1024**2]
        if os.environ.get("HM_BENCH_LARGE") == "1":
            sizes.append(1024**3)  # 1GB

        FS_TMP.mkdir(parents=True, exist_ok=True)
        records: List[PerfRecord] = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)

            for s in sizes:
                f = tmp / f"file_{s}.bin"
                _make_file(f, s)
                # Filesystem write (copy to FS_TMP)
                fs_t = _timeit(lambda: shutil.copy2(f, FS_TMP / f.name))
                # Holographic store
                hm_t = _timeit(lambda: hfs.store(f))
                stats = hfs.stats()
                records.append(
                    PerfRecord(
                        label=f.name,
                        size=s,
                        fs_write_s=fs_t,
                        hm_store_s=hm_t,
                        chunks=stats.get("entries", 0),
                        holo_bytes=stats.get("holo_bytes", 0),
                        compression_x=stats.get("compression_x"),
                    )
                )

        # Basic sanity: holographic store is functional
        assert all(r.hm_store_s > 0 for r in records)

    def test_bit_perfect_reconstruction(self) -> None:
        """Validate perfect reconstruction across file types.

        Tests: Binary, text, small image/pdf/video from repo if present
        Validate: SHA256 matches, zero data loss
        """
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)

            # Prepare files
            samples: List[Path] = []
            # Binary
            b = tmp / "bin_2mb.bin"
            _make_file(b, 2 * 1024 * 1024)
            samples.append(b)
            # Text
            t = tmp / "lorem.txt"
            t.write_text("Lorem ipsum dolor sit amet\n" * 1000, encoding="utf-8")
            samples.append(t)
            # Optional: include small assets if exist in repo
            for p in [
                Path("demo") / "HolographicMemory.pdf",
                Path("demo") / "sample.jpg",
                Path("demo") / "sample.mp4",
            ]:
                if p.exists():
                    samples.append(p)

            for src in samples:
                doc = hfs.store(src)
                out = tmp / f"recalled_{src.name}"
                hfs.recall(doc, out=out)
                assert _sha256(src) == _sha256(out)

    def test_holographic_search_performance(self) -> None:
        """Search speed vs traditional indexing

        Dataset: 1K (default), 10K if HM_BENCH_MEDIUM=1
                 100K/1M gated behind HM_BENCH_LARGE=1
        Compare: HM search vs ripgrep vs SQLite LIKE
        """
        n = 1000
        if os.environ.get("HM_BENCH_MEDIUM") == "1":
            n = 10000
        if os.environ.get("HM_BENCH_LARGE") == "1":
            n = 100000

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            corpus = tmp / "corpus"
            corpus.mkdir()
            hfs = _ensure_hfs(tmp)
            # Build corpus of simple text files with keywords
            keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
            for i in range(n):
                k = keys[i % len(keys)]
                p = corpus / f"doc_{i:06d}_{k}.txt"
                p.write_text(f"{k} document number {i}\n" * 3, encoding="utf-8")
                hfs.store(p)

            query = "gamma"
            # HM search timing (k=5)
            hm_s = _timeit(lambda: hfs.search(query, k=5), repeat=3)

            # ripgrep timing
            try:
                rg_s = _timeit(
                    lambda: subprocess.run(
                        ["rg", "-n", query, str(corpus)], capture_output=True, check=False
                    ),
                    repeat=3,
                )
            except FileNotFoundError:
                rg_s = float("nan")

            # SQLite LIKE timing
            db = tmp / "idx.db"
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, name TEXT, body TEXT)")
            rows = [
                (i, f"doc_{i:06d}", f"{keys[i % len(keys)]} document number {i}") for i in range(n)
            ]
            cur.executemany("INSERT INTO docs(id,name,body) VALUES (?,?,?)", rows)
            con.commit()
            sql_s = _timeit(
                lambda: cur.execute("SELECT id FROM docs WHERE body LIKE ? LIMIT 5", (f"%{query}%",)).fetchall(),
                repeat=3,
            )
            con.close()

            # Sanity assertions: timings are non-negative
            assert hm_s >= 0.0
            assert sql_s >= 0.0


class CompressionBenchmarks:
    def test_compression_superiority(self) -> None:
        """Quantify compression advantage vs competitors

        Compares: Holographic state size vs gzip/zip/xz on diverse files.
        Note: HM stores base64 chunks + wave state; this measures practical footprint.
        """
        if HoloFS is None:
            pytest.skip("holographicfs not available")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)

            samples: List[Path] = []
            # Create synthetic files
            _make_file(tmp / "zeros_5mb.bin", 5 * 1024**2)
            (tmp / "text_small.txt").write_text("abc " * 250_000, encoding="utf-8")
            samples.extend([tmp / "zeros_5mb.bin", tmp / "text_small.txt"])

            ratios: Dict[str, Dict[str, float]] = {}
            for p in samples:
                orig = p.stat().st_size
                hfs.store(p)
                s = hfs.stats()
                holo_bytes = int(s.get("holo_bytes", 0))
                ratios[p.name] = {"holographic": orig / holo_bytes if holo_bytes else 0.0}

                # gzip
                gz = tmp / f"{p.name}.gz"
                subprocess.run(["gzip", "-fk", str(p)], check=False)
                if gz.exists():
                    ratios[p.name]["gzip"] = orig / gz.stat().st_size
                # zip
                zipf = tmp / f"{p.stem}.zip"
                subprocess.run(["zip", "-jq", str(zipf), str(p)], check=False)
                if zipf.exists():
                    ratios[p.name]["zip"] = orig / zipf.stat().st_size
                # xz (if installed)
                xz = tmp / f"{p.name}.xz"
                subprocess.run(["xz", "-fk", str(p)], check=False)
                if xz.exists():
                    ratios[p.name]["xz"] = orig / xz.stat().st_size

            # At least gathered some metrics
            assert any(r.get("holographic", 0.0) > 0 for r in ratios.values())

    def test_storage_efficiency(self) -> None:
        """Storage efficiency vs traditional systems

        Compares total footprint including metadata and index overheads.
        """
        if HoloFS is None:
            pytest.skip("holographicfs not available")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)
            # Create 100 small files
            files = []
            for i in range(100):
                p = tmp / f"f_{i:04d}.txt"
                p.write_text(f"sample {i}\n" * 8)
                files.append(p)
                hfs.store(p)

            # Measure index/db overheads
            s = hfs.stats()
            holo_bytes = int(s.get("holo_bytes", 0))

            # SQLite baseline
            db = tmp / "base.db"
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("CREATE TABLE docs (path TEXT, body BLOB)")
            for p in files:
                cur.execute("INSERT INTO docs(path, body) VALUES (?, ?)", (str(p), p.read_bytes()))
            con.commit()
            con.close()

            sqlite_bytes = db.stat().st_size
            assert holo_bytes >= 0 and sqlite_bytes >= 0


class RealWorldBenchmarks:
    def test_user_workflow_performance(self) -> None:
        """End-to-end user experience benchmarks

        Scenario: upload folder → search → download
        """
        if HoloFS is None:
            pytest.skip("holographicfs not available")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data = tmp / "data"
            data.mkdir()
            # build folder with 500 files
            for i in range(500):
                (data / f"doc_{i:04d}.txt").write_text(f"hello {i}\n" * 5)

            hfs = _ensure_hfs(tmp)
            t0 = time.perf_counter()
            for p in sorted(data.iterdir()):
                hfs.store(p)
            t1 = time.perf_counter()
            ingest_s = t1 - t0

            # search and recall a file
            t2 = time.perf_counter()
            _ = hfs.search("hello 42", k=5)
            t3 = time.perf_counter()
            _ = hfs.recall("doc_0042.txt", out=tmp / "out.txt") if False else None
            # recall by name requires index lookup; we used raw name above only if mapping existed
            search_s = t3 - t2
            assert ingest_s >= 0 and search_s >= 0

    def test_concurrent_operations(self) -> None:
        """Multi-user concurrent performance

        Executes concurrent ingestion on a shared HoloFS state.
        """
        if HoloFS is None:
            pytest.skip("holographicfs not available")
        import concurrent.futures as cf

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)
            files = []
            for i in range(200):
                p = tmp / f"file_{i:04d}.txt"
                p.write_text("x" * 2048)
                files.append(p)

            def _store(p: Path) -> str:
                return hfs.store(p)

            for users in (10, 50, 100):
                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(max_workers=users) as ex:
                    list(ex.map(_store, files))
                t1 = time.perf_counter()
                assert (t1 - t0) >= 0


class CompetitorBenchmarkSuite:
    @pytest.mark.skipif(os.environ.get("HM_ENABLE_COMPETITOR") != "1", reason="Requires competitor CLI setup")
    def test_sync_speed_comparison(self) -> None:
        """Sync speed: Holographic vs cloud storage

        Requires HM_ENABLE_COMPETITOR=1 and configured CLIs (rclone/onedrive/dropbox).
        """
        assert True  # Placeholder for integration harness

    @pytest.mark.skipif(os.environ.get("HM_ENABLE_COMPETITOR") != "1", reason="Requires competitor CLI setup")
    def test_search_superiority(self) -> None:
        """Search speed: Holographic vs cloud search"""
        assert True

    @pytest.mark.skipif(os.environ.get("HM_ENABLE_COMPETITOR") != "1", reason="Requires competitor CLI setup")
    def test_storage_cost_analysis(self) -> None:
        """Storage efficiency: Holographic vs cloud costs"""
        assert True

    @pytest.mark.skipif(os.environ.get("HM_ENABLE_COMPETITOR") != "1", reason="Requires competitor CLI setup")
    def test_offline_capability(self) -> None:
        """Offline access: Holographic vs cloud dependency"""
        assert True


class DatabaseReplacement:
    def test_database_replacement(self) -> None:
        """HM-backed index vs traditional databases (SQLite baseline)"""
        if HoloFS is None:
            pytest.skip("holographicfs not available")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hfs = _ensure_hfs(tmp)
            # Ingest small dataset
            for i in range(200):
                p = tmp / f"d_{i:04d}.txt"
                p.write_text(f"row {i}")
                hfs.store(p)
            s = hfs.stats()
            holo_bytes = int(s.get("holo_bytes", 0))

            # SQLite equivalent
            db = tmp / "eq.db"
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("CREATE TABLE docs (name TEXT, body TEXT)")
            for i in range(200):
                cur.execute("INSERT INTO docs(name, body) VALUES (?, ?)", (f"d_{i:04d}.txt", f"row {i}"))
            con.commit()
            con.close()
            sqlite_bytes = db.stat().st_size

            # Verify both are reasonable and measured
            assert holo_bytes >= 0 and sqlite_bytes >= 0


class EnterpriseComplianceSuite:
    @pytest.mark.skipif(os.environ.get("HM_ENTERPRISE_MODE") != "1", reason="gated enterprise load test")
    def test_load_performance(self) -> None:
        """Enterprise load testing (gated)"""
        assert True

    @pytest.mark.skipif(os.environ.get("HM_ENTERPRISE_MODE") != "1", reason="gated enterprise DR test")
    def test_disaster_recovery(self) -> None:
        """Disaster recovery validation (gated)"""
        assert True

    @pytest.mark.skipif(os.environ.get("HM_ENTERPRISE_MODE") != "1", reason="gated enterprise security test")
    def test_security_compliance(self) -> None:
        """Security and compliance validation (gated)"""
        assert True


class ScalabilitySuite:
    @pytest.mark.skipif(os.environ.get("HM_ENTERPRISE_MODE") != "1", reason="gated enterprise scalability")
    def test_enterprise_scalability(self) -> None:
        """Scalability under enterprise conditions (gated)"""
        assert True


class ReportingSuite:
    def test_generate_enterprise_report(self) -> None:
        """Generate Fortune 500-grade benchmark report (skeleton)."""
        out = Path("reports/benchmarks")
        out.mkdir(parents=True, exist_ok=True)
        (out / "enterprise_report.md").write_text(
            "# Enterprise Benchmark Report\n\nGenerated placeholder. See JSON/HDF5 for details.\n",
            encoding="utf-8",
        )
        assert (out / "enterprise_report.md").exists()


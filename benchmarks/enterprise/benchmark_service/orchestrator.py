from __future__ import annotations

import platform
import time
import uuid
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from .adapters import FaissIndex, HoloBackend
from .contracts import BenchmarkResult, DataSpec, Metric, PipelineResult, RunReport
from .pipelines import EmbeddingStoragePipeline
from .services import MetricsService, ReportingService, TestDataService, TheoryService


class BenchmarkService:
    """
    Core orchestration for benchmark categories.
    """

    def __init__(self, spec: DataSpec) -> None:
        self.spec = spec
        self.data = TestDataService(spec).generate()
        self.backend = HoloBackend(vector_dim=spec.vector_dim, field_shape=spec.field_shape)
        self.metrics = MetricsService()
        self.theory = TheoryService(n=spec.vector_dim)
        self.reporting = ReportingService()

    # -----------------------------
    # Categories
    # -----------------------------
    def bench_storage(self) -> BenchmarkResult:
        items = {t.id: t.text.encode("utf-8") for t in self.data.texts}
        st = self.backend.store_many(items)
        m = [
            Metric(name="stored_count", value=float(st.count)),
            Metric(name="duration_ms", value=st.duration_ms, unit="ms"),
            Metric(name="throughput", value=st.throughput_items_per_s, unit="items/s"),
        ]
        return BenchmarkResult(category="storage", name="holo_store", metrics=m)

    def bench_retrieval_accuracy(self) -> BenchmarkResult:
        # Build resonance index and evaluate top-1 success on seen items
        self.backend.index_vectors({t.id: self.data.states[t.id] for t in self.data.texts})
        ok = 0
        for t in self.data.texts:
            res = self.backend.query_vectors(self.data.states[t.id], k=1)
            if res and res[0][0] == t.id:
                ok += 1
        rate = ok / max(1, len(self.data.texts))
        m = [Metric(name="top1_success_rate", value=rate)]
        return BenchmarkResult(category="retrieval", name="holo_resonance_top1", metrics=m)

    def bench_fft_ops(self) -> BenchmarkResult:
        # 3D FFT roundtrip and timing
        if not self.data.fields3d:
            f = np.zeros(self.spec.field_shape, dtype=complex)
        else:
            f = next(iter(self.data.fields3d.values()))
        def _fft_ifft() -> None:
            F = self.backend.fft3d(f)
            _ = self.backend.ifft3d(F)
        t = self.metrics.time_many(_fft_ifft, n=5)
        # small n unitarity error estimate (via theory)
        tol = self.theory.predictions().fft_unitarity_err_tol
        m = [
            Metric(name="fft3d_p50_ms", value=t.p50_ms, unit="ms"),
            Metric(name="fft3d_p95_ms", value=t.p95_ms, unit="ms"),
            Metric(name="fft_unitarity_tol", value=tol),
        ]
        return BenchmarkResult(category="fft", name="fft3d_roundtrip", metrics=m)

    def bench_faiss_comparison(self) -> BenchmarkResult:
        # Build FAISS over our complex embeddings (converted to real)
        faiss = FaissIndex(dim_complex=self.spec.vector_dim)
        info = faiss.info()
        if not info.available:
            return BenchmarkResult(
                category="comparison",
                name="faiss_vs_holo",
                metrics=[Metric(name="faiss_available", value=0.0)],
                notes="FAISS not available",
            )
        # map id order to index
        ids = [t.id for t in self.data.texts]
        vecs = [self.data.states[i] for i in ids]
        faiss.add(vecs)
        # Evaluate top1 agreement between FAISS and resonance index
        self.backend.index_vectors({i: v for i, v in zip(ids, vecs)})
        agree = 0
        for idx, _id in enumerate(ids[: min(50, len(ids))]):
            q = self.data.states[_id]
            f_top = faiss.search(q, k=1)
            h_top = self.backend.query_vectors(q, k=1)
            faiss_id = ids[f_top[0][0]] if f_top and f_top[0][0] >= 0 else None
            holo_id = h_top[0][0] if h_top else None
            if faiss_id == holo_id:
                agree += 1
        rate = agree / max(1, min(50, len(ids)))
        m = [
            Metric(name="faiss_available", value=1.0),
            Metric(name="top1_agreement_rate", value=rate),
        ]
        return BenchmarkResult(category="comparison", name="faiss_vs_holo", metrics=m)

    def pipeline_embeddings_storage_retrieval(self) -> PipelineResult:
        id_to_text = {t.id: t.text for t in self.data.texts}
        id_to_vec = {t.id: self.data.states[t.id] for t in self.data.texts}
        queries = [t.id for t in self.data.texts[: min(50, len(self.data.texts))]]
        pipe = EmbeddingStoragePipeline(vector_dim=self.spec.vector_dim)
        stats = pipe.run(id_to_text, id_to_vec, queries)
        return PipelineResult(
            pipeline_name="embeddings→store→retrieve→text",
            success_rate=float(stats["query_success_rate"]),
            latency_ms_p50=float(stats["query_p50_ms"]),
            latency_ms_p95=float(stats["query_p95_ms"]),
            details=stats,
        )

    # -----------------------------
    # End-to-end run
    # -----------------------------
    def run_all(self) -> RunReport:
        rid = uuid.uuid4().hex[:8]
        results: List[BenchmarkResult] = []
        results.append(self.bench_storage())
        results.append(self.bench_retrieval_accuracy())
        results.append(self.bench_fft_ops())
        results.append(self.bench_faiss_comparison())
        pipeline = self.pipeline_embeddings_storage_retrieval()

        report = RunReport(
            run_id=rid,
            platform=platform.platform(),
            dataset=self.spec,
            results=results,
            pipelines=[pipeline],
        )
        # Persist
        self.reporting.write_json(report)
        # Optional: store a sample field spectrum
        arrays: Dict[str, np.ndarray] = {}
        if self.data.fields3d:
            any_field = next(iter(self.data.fields3d.values()))
            arrays["sample_field3d_mag"] = np.abs(any_field).astype("float32")
        self.reporting.write_hdf5(arrays, run_id=rid)
        return report


if __name__ == "__main__":  # Manual local run
    spec = DataSpec(name="default", num_items=64, vector_dim=256, field_shape=(32, 32, 32))
    svc = BenchmarkService(spec)
    rep = svc.run_all()
    print(f"Benchmark run complete: {rep.run_id}")


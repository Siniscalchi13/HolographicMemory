import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.enterprise.holographic_benchmarks import (
    HolographicBenchmarkSuite,
    CompressionBenchmarks,
    RealWorldBenchmarks,
    CompetitorBenchmarkSuite,
    DatabaseReplacement,
    EnterpriseComplianceSuite,
    ScalabilitySuite,
    ReportingSuite,
)


class TestHolographicBenchmarks:
    def test_chunked_storage_performance(self):
        HolographicBenchmarkSuite().test_chunked_storage_performance()

    def test_bit_perfect_reconstruction(self):
        HolographicBenchmarkSuite().test_bit_perfect_reconstruction()

    def test_holographic_search_performance(self):
        HolographicBenchmarkSuite().test_holographic_search_performance()


class TestCompression:
    def test_compression_superiority(self):
        CompressionBenchmarks().test_compression_superiority()

    def test_storage_efficiency(self):
        CompressionBenchmarks().test_storage_efficiency()


class TestRealWorld:
    def test_user_workflow_performance(self):
        RealWorldBenchmarks().test_user_workflow_performance()

    def test_concurrent_operations(self):
        RealWorldBenchmarks().test_concurrent_operations()


class TestCompetitors:
    def test_sync_speed_comparison(self):
        CompetitorBenchmarkSuite().test_sync_speed_comparison()

    def test_search_superiority(self):
        CompetitorBenchmarkSuite().test_search_superiority()

    def test_storage_cost_analysis(self):
        CompetitorBenchmarkSuite().test_storage_cost_analysis()

    def test_offline_capability(self):
        CompetitorBenchmarkSuite().test_offline_capability()


class TestDBReplacement:
    def test_database_replacement(self):
        DatabaseReplacement().test_database_replacement()


class TestEnterpriseCompliance:
    def test_load_performance(self):
        EnterpriseComplianceSuite().test_load_performance()

    def test_disaster_recovery(self):
        EnterpriseComplianceSuite().test_disaster_recovery()

    def test_security_compliance(self):
        EnterpriseComplianceSuite().test_security_compliance()


class TestScalability:
    def test_enterprise_scalability(self):
        ScalabilitySuite().test_enterprise_scalability()


class TestReporting:
    def test_generate_enterprise_report(self):
        ReportingSuite().test_generate_enterprise_report()

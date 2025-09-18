#!/usr/bin/env python3
"""
Unit tests for performance benchmark modules
"""

import unittest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_benchmark import MemoryBenchmark
from cpu_benchmark import CPUBenchmark
from storage_benchmark import StorageBenchmark
from performance_report import PerformanceReportGenerator, combine_benchmark_results


class TestMemoryBenchmark(unittest.TestCase):
    """Test memory benchmark module"""

    def setUp(self):
        self.benchmark = MemoryBenchmark()

    def test_bandwidth_test_small(self):
        """Test small memory bandwidth measurement"""
        result = self.benchmark.test_bandwidth(size_mb=1, iterations=2)
        self.assertIn('avg_bandwidth_gbps', result)
        self.assertGreater(result['avg_bandwidth_gbps'], 0)

    def test_cache_level_analysis(self):
        """Test cache level detection"""
        # Create mock test results
        tests = [
            {'size_mb': 1, 'avg_bandwidth_gbps': 100},
            {'size_mb': 8, 'avg_bandwidth_gbps': 80},
            {'size_mb': 32, 'avg_bandwidth_gbps': 40},
            {'size_mb': 256, 'avg_bandwidth_gbps': 20}
        ]
        analysis = self.benchmark._analyze_cache_levels(tests)
        self.assertIn('ram_bandwidth', analysis)
        self.assertIsNotNone(analysis['ram_bandwidth'])


class TestCPUBenchmark(unittest.TestCase):
    """Test CPU benchmark module"""

    def setUp(self):
        self.benchmark = CPUBenchmark()

    def test_cpu_info_extraction(self):
        """Test CPU information extraction"""
        info = self.benchmark._get_cpu_info()
        self.assertIn('physical_cores', info)
        self.assertIn('logical_cores', info)
        self.assertIn('model', info)
        self.assertGreater(info['logical_cores'], 0)

    def test_matrix_multiplication(self):
        """Test matrix multiplication benchmark"""
        result = self.benchmark.matrix_multiplication_test(size=100, iterations=2)
        self.assertIn('avg_gflops', result)
        self.assertGreater(result['avg_gflops'], 0)

    def test_prime_calculation(self):
        """Test prime calculation benchmark"""
        result = self.benchmark.prime_calculation_test(limit=1000)
        self.assertIn('primes_found', result)
        self.assertIn('primes_per_sec', result)
        self.assertGreater(result['primes_found'], 0)

    def test_hash_benchmark(self):
        """Test hash calculation benchmark"""
        result = self.benchmark.hash_benchmark(size_mb=1, iterations=2)
        self.assertIn('sha256', result)
        self.assertIn('throughput_mbps', result['sha256'])
        self.assertGreater(result['sha256']['throughput_mbps'], 0)


class TestStorageBenchmark(unittest.TestCase):
    """Test storage benchmark module"""

    def setUp(self):
        self.benchmark = StorageBenchmark()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_storage_type_detection(self):
        """Test storage type detection"""
        storage_type = self.benchmark._detect_storage_type(Path(self.temp_dir))
        self.assertIn(storage_type, ['SSD', 'HDD', 'Unknown'])

    def test_sequential_read(self):
        """Test sequential read benchmark"""
        result = self.benchmark.sequential_read_test(
            Path(self.temp_dir),
            size_mb=10,
            block_size_kb=1024
        )
        if 'error' not in result:
            self.assertIn('read_mbps', result)
            self.assertGreater(result['read_mbps'], 0)

    def test_random_read_iops(self):
        """Test random read IOPS measurement"""
        result = self.benchmark.random_read_test(
            Path(self.temp_dir),
            file_size_mb=1,
            read_count=100,
            block_size_kb=4
        )
        if 'error' not in result:
            self.assertIn('iops', result)
            self.assertGreater(result['iops'], 0)


class TestPerformanceReport(unittest.TestCase):
    """Test performance report generator"""

    def setUp(self):
        self.generator = PerformanceReportGenerator()

    def test_add_results(self):
        """Test adding benchmark results"""
        test_data = {'test': 'data', 'value': 123}
        self.generator.add_results('memory', test_data)
        self.assertIn('memory', self.generator.results)
        self.assertEqual(self.generator.results['memory'], test_data)

    def test_json_report_generation(self):
        """Test JSON report generation"""
        self.generator.add_results('memory', {
            'analysis': {'ram_bandwidth': 50},
            'system_memory_gb': 32
        })
        report = self.generator.generate_json_report()
        self.assertIn('timestamp', report)
        self.assertIn('system_info', report)
        self.assertIn('benchmarks', report)
        self.assertIn('summary', report)
        self.assertIn('recommendations', report)

    def test_performance_scoring(self):
        """Test performance score calculation"""
        self.generator.add_results('memory', {
            'analysis': {'ram_bandwidth': 50}
        })
        self.generator.add_results('cpu', {
            'tests': {'matrix_mult': {'avg_gflops': 100}}
        })
        summary = self.generator._generate_summary()
        self.assertIn('performance_score', summary)
        self.assertGreater(summary['performance_score'], 0)
        self.assertIn('overall_rating', summary)

    def test_recommendations_generation(self):
        """Test recommendation generation"""
        self.generator.add_results('memory', {
            'analysis': {'ram_bandwidth': 15},
            'system_memory_gb': 16
        })
        recommendations = self.generator._generate_recommendations()
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_console_report(self):
        """Test console report generation"""
        self.generator.add_results('cpu', {
            'cpu_info': {'model': 'Test CPU', 'physical_cores': 8},
            'tests': {'matrix_mult': {'avg_gflops': 75}}
        })
        report = self.generator.generate_console_report()
        self.assertIn('PERFORMANCE REPORT', report)
        self.assertIn('Test CPU', report)
        self.assertIn('GFLOPS', report)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_combine_benchmark_results(self):
        """Test combining results from multiple benchmarks"""
        memory_results = {
            'analysis': {'ram_bandwidth': 40},
            'system_memory_gb': 64
        }
        cpu_results = {
            'cpu_info': {'physical_cores': 16},
            'tests': {'matrix_mult': {'avg_gflops': 120}}
        }
        storage_results = {
            'summary': {'best_speed_mbps': 2000}
        }

        generator = combine_benchmark_results(
            memory_results=memory_results,
            cpu_results=cpu_results,
            storage_results=storage_results
        )

        self.assertIn('memory', generator.results)
        self.assertIn('cpu', generator.results)
        self.assertIn('storage', generator.results)

        report = generator.generate_json_report()
        self.assertEqual(report['summary']['overall_rating'], 'Good')


def run_quick_tests():
    """Run a quick subset of tests"""
    suite = unittest.TestSuite()

    # Add quick tests only
    suite.addTest(TestMemoryBenchmark('test_bandwidth_test_small'))
    suite.addTest(TestCPUBenchmark('test_cpu_info_extraction'))
    suite.addTest(TestStorageBenchmark('test_storage_type_detection'))
    suite.addTest(TestPerformanceReport('test_add_results'))
    suite.addTest(TestPerformanceReport('test_performance_scoring'))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    # Run quick tests by default, full suite with --full flag
    if '--full' in sys.argv:
        unittest.main(argv=sys.argv[:1])  # Remove custom args
    else:
        print("Running quick tests (use --full for complete suite)...")
        result = run_quick_tests()
        sys.exit(0 if result.wasSuccessful() else 1)
#!/usr/bin/env python3
"""
Memory Performance Benchmark Module
Tests RAM bandwidth with various block sizes and access patterns
"""

import time
import numpy as np
import psutil
from typing import Dict, List, Tuple
import gc
import json

# Import configuration
try:
    from config import BENCHMARK_CONFIG
    MEM_CONFIG = BENCHMARK_CONFIG.get('memory', {})
except ImportError:
    # Fallback if config not available
    MEM_CONFIG = {}


class MemoryBenchmark:
    """Memory bandwidth testing utilities"""

    def __init__(self):
        self.results = {}
        self.system_memory = psutil.virtual_memory().total / (1024**3)  # GB

    def test_bandwidth(self, size_mb: int, iterations: int = None, pattern: str = "sequential") -> Dict[str, float]:
        if iterations is None:
            iterations = MEM_CONFIG.get('iterations', 5)
        """Test memory bandwidth for a given size"""
        size_bytes = size_mb * 1024 * 1024
        results = {
            'size_mb': size_mb,
            'pattern': pattern,
            'iterations': iterations,
            'times': [],
            'bandwidth_gbps': []
        }

        # Create test array
        try:
            if pattern == "sequential":
                test_array = np.zeros(size_bytes // 8, dtype=np.float64)
            else:  # random
                test_array = np.random.rand(size_bytes // 8).astype(np.float64)

            # Warm up
            _ = test_array.copy()
            gc.collect()

            # Run tests
            for _ in range(iterations):
                gc.collect()
                start = time.perf_counter()

                if pattern == "sequential":
                    result = test_array.copy()
                else:  # random access simulation
                    indices = np.random.randint(0, len(test_array), size=len(test_array)//10)
                    result = test_array[indices].copy()

                elapsed = time.perf_counter() - start

                # Calculate bandwidth
                if pattern == "sequential":
                    data_gb = (size_bytes * 2) / (1024**3)  # Read + Write
                else:
                    data_gb = (len(indices) * 8 * 2) / (1024**3)

                bandwidth = data_gb / elapsed if elapsed > 0 else 0

                results['times'].append(elapsed)
                results['bandwidth_gbps'].append(bandwidth)

                del result

            # Calculate statistics
            results['avg_bandwidth_gbps'] = np.mean(results['bandwidth_gbps'])
            results['std_bandwidth_gbps'] = np.std(results['bandwidth_gbps'])
            results['min_bandwidth_gbps'] = np.min(results['bandwidth_gbps'])
            results['max_bandwidth_gbps'] = np.max(results['bandwidth_gbps'])

            # Clean up
            del test_array
            gc.collect()

        except MemoryError:
            results['error'] = f"Not enough memory for {size_mb}MB test"
            results['avg_bandwidth_gbps'] = 0

        return results

    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run complete memory benchmark suite"""
        print("\n🧪 Memory Performance Testing")
        print("=" * 50)

        # Define test sizes based on available memory and config
        test_sizes = []

        # Small (cache tests)
        test_sizes.extend(MEM_CONFIG.get('test_sizes_mb', [1, 4, 8]))

        # Medium (L3 cache and RAM transition)
        min_mem_medium = MEM_CONFIG.get('min_system_memory_gb', {}).get('medium', 8)
        if self.system_memory > min_mem_medium:
            test_sizes.extend(MEM_CONFIG.get('test_sizes_mb_medium', [32, 64, 128]))

        # Large (pure RAM)
        min_mem_large = MEM_CONFIG.get('min_system_memory_gb', {}).get('large', 16)
        if self.system_memory > min_mem_large:
            test_sizes.extend(MEM_CONFIG.get('test_sizes_mb_large', [256, 512, 1024]))

        min_mem_xlarge = MEM_CONFIG.get('min_system_memory_gb', {}).get('xlarge', 32)
        if self.system_memory > min_mem_xlarge:
            test_sizes.extend(MEM_CONFIG.get('test_sizes_mb_xlarge', [2048, 4096]))

        results = {
            'system_memory_gb': self.system_memory,
            'sequential_tests': [],
            'random_tests': []
        }

        # Sequential access tests
        print("\n📊 Sequential Access Pattern:")
        for size_mb in test_sizes:
            print(f"  Testing {size_mb}MB... ", end='', flush=True)
            iterations = MEM_CONFIG.get('iterations', 3)
            result = self.test_bandwidth(size_mb, iterations=iterations, pattern="sequential")
            results['sequential_tests'].append(result)
            if 'error' not in result:
                print(f"✓ {result['avg_bandwidth_gbps']:.2f} GB/s")
            else:
                print(f"✗ {result['error']}")

        # Random access tests (smaller sizes only)
        print("\n🎲 Random Access Pattern:")
        for size_mb in test_sizes[:5]:  # Only test smaller sizes for random
            print(f"  Testing {size_mb}MB... ", end='', flush=True)
            iterations = MEM_CONFIG.get('iterations', 3)
            result = self.test_bandwidth(size_mb, iterations=iterations, pattern="random")
            results['random_tests'].append(result)
            if 'error' not in result:
                print(f"✓ {result['avg_bandwidth_gbps']:.2f} GB/s")
            else:
                print(f"✗ {result['error']}")

        # Analyze cache levels
        results['analysis'] = self._analyze_cache_levels(results['sequential_tests'])

        return results

    def _analyze_cache_levels(self, sequential_tests: List[Dict]) -> Dict[str, any]:
        """Analyze results to identify cache levels"""
        analysis = {
            'l1_l2_bandwidth': None,
            'l3_bandwidth': None,
            'ram_bandwidth': None,
            'cache_size_estimates': {}
        }

        if not sequential_tests:
            return analysis

        # Sort by size
        sorted_tests = sorted(sequential_tests, key=lambda x: x['size_mb'])
        bandwidths = [(t['size_mb'], t.get('avg_bandwidth_gbps', 0)) for t in sorted_tests]

        # Simple heuristic: significant drops indicate cache boundaries
        for i in range(1, len(bandwidths)):
            prev_bw = bandwidths[i-1][1]
            curr_bw = bandwidths[i][1]
            size_mb = bandwidths[i][0]

            if prev_bw > 0 and curr_bw > 0:
                drop_ratio = curr_bw / prev_bw

                if drop_ratio < 0.7:  # 30% or more drop
                    if not analysis['l1_l2_bandwidth'] and size_mb <= 16:
                        analysis['l1_l2_bandwidth'] = prev_bw
                        analysis['cache_size_estimates']['l1_l2_mb'] = bandwidths[i-1][0]
                    elif not analysis['l3_bandwidth'] and size_mb <= 256:
                        analysis['l3_bandwidth'] = prev_bw
                        analysis['cache_size_estimates']['l3_mb'] = bandwidths[i-1][0]

        # RAM bandwidth is typically the last stable value
        if bandwidths:
            analysis['ram_bandwidth'] = bandwidths[-1][1]

        return analysis

    def generate_report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("\n" + "="*60)
        report.append("📊 MEMORY PERFORMANCE REPORT")
        report.append("="*60)
        report.append(f"System Memory: {results['system_memory_gb']:.1f} GB")

        if results.get('analysis'):
            analysis = results['analysis']
            report.append("\n🔍 Detected Memory Hierarchy:")
            if analysis['l1_l2_bandwidth']:
                report.append(f"  L1/L2 Cache: ~{analysis['l1_l2_bandwidth']:.1f} GB/s")
            if analysis['l3_bandwidth']:
                report.append(f"  L3 Cache: ~{analysis['l3_bandwidth']:.1f} GB/s")
            if analysis['ram_bandwidth']:
                report.append(f"  RAM: ~{analysis['ram_bandwidth']:.1f} GB/s")

        report.append("\n📈 Performance Summary:")
        report.append("  Sequential Access:")
        for test in results['sequential_tests'][-3:]:  # Show last 3 results
            if 'error' not in test:
                report.append(f"    {test['size_mb']:4d}MB: {test['avg_bandwidth_gbps']:6.2f} GB/s "
                            f"(±{test['std_bandwidth_gbps']:.2f})")

        report.append("\n  Random Access:")
        for test in results.get('random_tests', [])[:3]:
            if 'error' not in test:
                report.append(f"    {test['size_mb']:4d}MB: {test['avg_bandwidth_gbps']:6.2f} GB/s")

        return "\n".join(report)


def main():
    """Run memory benchmark standalone"""
    benchmark = MemoryBenchmark()
    results = benchmark.run_comprehensive_test()
    print(benchmark.generate_report(results))

    # Save JSON results
    with open('memory_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n💾 Results saved to memory_benchmark_results.json")


if __name__ == "__main__":
    main()
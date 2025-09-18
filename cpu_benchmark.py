#!/usr/bin/env python3
"""
CPU Performance Benchmark Module
Tests CPU performance with various workloads and detects features
"""

import time
import numpy as np
import hashlib
import multiprocessing
import subprocess
import platform
import json
import os
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class CPUBenchmark:
    """CPU performance testing utilities"""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.cpu_info = self._get_cpu_info()
        self.results = {}

    def _get_cpu_info(self) -> Dict[str, any]:
        """Extract CPU information from system"""
        info = {
            'physical_cores': multiprocessing.cpu_count(),
            'logical_cores': multiprocessing.cpu_count(),
            'model': 'Unknown',
            'vendor': 'Unknown',
            'features': [],
            'bogomips': None,
            'frequency': {}
        }

        # Try to get info from /proc/cpuinfo (Linux)
        if os.path.exists('/proc/cpuinfo'):
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()

                # Parse CPU model
                for line in cpuinfo.split('\n'):
                    if line.startswith('model name'):
                        info['model'] = line.split(':')[1].strip()
                        break

                # Parse vendor
                for line in cpuinfo.split('\n'):
                    if line.startswith('vendor_id'):
                        info['vendor'] = line.split(':')[1].strip()
                        break

                # Parse BogoMIPS
                for line in cpuinfo.split('\n'):
                    if line.startswith('bogomips'):
                        try:
                            info['bogomips'] = float(line.split(':')[1].strip())
                        except:
                            pass
                        break

                # Parse CPU flags for features
                for line in cpuinfo.split('\n'):
                    if line.startswith('flags'):
                        flags = line.split(':')[1].strip().split()
                        # Check for important instruction sets
                        for feature in ['avx', 'avx2', 'avx512f', 'sse4_1', 'sse4_2', 'fma', 'f16c']:
                            if feature in flags:
                                info['features'].append(feature.upper())
                        break

                # Count physical vs logical cores
                physical_ids = set()
                for line in cpuinfo.split('\n'):
                    if line.startswith('physical id'):
                        physical_ids.add(line.split(':')[1].strip())
                if physical_ids:
                    info['physical_cores'] = len(physical_ids) * (info['logical_cores'] // max(1, len(physical_ids)))

            except Exception as e:
                print(f"Warning: Could not parse /proc/cpuinfo: {e}")

        # Try lscpu for additional info
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU MHz' in line:
                        try:
                            info['frequency']['current'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'CPU max MHz' in line:
                        try:
                            info['frequency']['max'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'CPU min MHz' in line:
                        try:
                            info['frequency']['min'] = float(line.split(':')[1].strip())
                        except:
                            pass
        except:
            pass

        return info

    def matrix_multiplication_test(self, size: int = 1000, iterations: int = 3) -> Dict[str, float]:
        """Benchmark matrix multiplication"""
        results = {
            'size': size,
            'iterations': iterations,
            'times': [],
            'gflops': []
        }

        # Create random matrices
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        # Warm up
        _ = np.matmul(A, B)

        for _ in range(iterations):
            start = time.perf_counter()
            C = np.matmul(A, B)
            elapsed = time.perf_counter() - start

            # Calculate GFLOPS (2*n^3 operations)
            ops = 2 * (size ** 3)
            gflops = (ops / elapsed) / 1e9

            results['times'].append(elapsed)
            results['gflops'].append(gflops)

        results['avg_gflops'] = np.mean(results['gflops'])
        results['std_gflops'] = np.std(results['gflops'])

        return results

    def prime_calculation_test(self, limit: int = 100000) -> Dict[str, float]:
        """Benchmark prime number calculation"""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        start = time.perf_counter()
        primes = [n for n in range(2, limit) if is_prime(n)]
        elapsed = time.perf_counter() - start

        return {
            'limit': limit,
            'primes_found': len(primes),
            'time': elapsed,
            'primes_per_sec': len(primes) / elapsed if elapsed > 0 else 0
        }

    def hash_benchmark(self, size_mb: int = 100, iterations: int = 5) -> Dict[str, float]:
        """Benchmark hash calculation performance"""
        data = os.urandom(size_mb * 1024 * 1024)
        algorithms = ['sha256', 'sha512', 'md5']
        results = {}

        for algo in algorithms:
            times = []
            for _ in range(iterations):
                hasher = hashlib.new(algo)
                start = time.perf_counter()
                hasher.update(data)
                _ = hasher.hexdigest()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            throughput = (size_mb / avg_time) if avg_time > 0 else 0

            results[algo] = {
                'avg_time': avg_time,
                'throughput_mbps': throughput,
                'std_time': np.std(times)
            }

        return results

    def multithread_test(self, workload='matrix', threads=None) -> Dict[str, float]:
        """Test multi-threaded vs single-threaded performance"""
        if threads is None:
            threads = self.cpu_count

        def worker_matrix():
            size = 500
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            return np.matmul(A, B).sum()

        def worker_prime():
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            return sum(1 for n in range(10000, 20000) if is_prime(n))

        worker = worker_matrix if workload == 'matrix' else worker_prime
        results = {}

        # Single-threaded test
        start = time.perf_counter()
        for _ in range(threads):
            worker()
        single_time = time.perf_counter() - start
        results['single_threaded'] = single_time

        # Multi-threaded test
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker) for _ in range(threads)]
            for future in futures:
                future.result()
        multi_time = time.perf_counter() - start
        results['multi_threaded'] = multi_time

        # Multi-process test
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker) for _ in range(threads)]
            for future in futures:
                future.result()
        process_time = time.perf_counter() - start
        results['multi_process'] = process_time

        results['speedup_threads'] = single_time / multi_time if multi_time > 0 else 1
        results['speedup_process'] = single_time / process_time if process_time > 0 else 1
        results['efficiency_threads'] = results['speedup_threads'] / threads
        results['efficiency_process'] = results['speedup_process'] / threads

        return results

    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run complete CPU benchmark suite"""
        print("\n🖥️ CPU Performance Testing")
        print("=" * 50)
        print(f"CPU: {self.cpu_info['model']}")
        print(f"Cores: {self.cpu_info['physical_cores']} physical, {self.cpu_info['logical_cores']} logical")
        if self.cpu_info['features']:
            print(f"Features: {', '.join(self.cpu_info['features'])}")

        results = {
            'cpu_info': self.cpu_info,
            'tests': {}
        }

        # Matrix multiplication test
        print("\n📊 Matrix Multiplication (1000x1000):")
        matrix_result = self.matrix_multiplication_test(size=1000, iterations=3)
        results['tests']['matrix_mult'] = matrix_result
        print(f"  Performance: {matrix_result['avg_gflops']:.2f} GFLOPS (±{matrix_result['std_gflops']:.2f})")

        # Prime calculation test
        print("\n🔢 Prime Number Calculation (up to 100k):")
        prime_result = self.prime_calculation_test(limit=100000)
        results['tests']['prime_calc'] = prime_result
        print(f"  Found {prime_result['primes_found']} primes in {prime_result['time']:.2f}s")
        print(f"  Rate: {prime_result['primes_per_sec']:.0f} primes/sec")

        # Hash benchmark
        print("\n🔐 Hash Calculation (100MB):")
        hash_result = self.hash_benchmark(size_mb=100, iterations=3)
        results['tests']['hash'] = hash_result
        for algo, perf in hash_result.items():
            print(f"  {algo.upper()}: {perf['throughput_mbps']:.1f} MB/s")

        # Multi-threading test
        print(f"\n🧵 Threading Efficiency ({self.cpu_count} cores):")
        thread_result = self.multithread_test(workload='matrix', threads=self.cpu_count)
        results['tests']['threading'] = thread_result
        print(f"  Thread speedup: {thread_result['speedup_threads']:.2f}x "
              f"(efficiency: {thread_result['efficiency_threads']*100:.1f}%)")
        print(f"  Process speedup: {thread_result['speedup_process']:.2f}x "
              f"(efficiency: {thread_result['efficiency_process']*100:.1f}%)")

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("\n" + "="*60)
        report.append("🖥️ CPU PERFORMANCE REPORT")
        report.append("="*60)

        info = results['cpu_info']
        report.append(f"CPU Model: {info['model']}")
        report.append(f"Cores: {info['physical_cores']}P/{info['logical_cores']}L")
        if info['bogomips']:
            report.append(f"BogoMIPS: {info['bogomips']:.2f}")
        if info['features']:
            report.append(f"SIMD Support: {', '.join(info['features'])}")

        tests = results.get('tests', {})

        if 'matrix_mult' in tests:
            mm = tests['matrix_mult']
            report.append(f"\n📊 Compute Performance:")
            report.append(f"  Matrix Multiply: {mm['avg_gflops']:.2f} GFLOPS")

        if 'prime_calc' in tests:
            pc = tests['prime_calc']
            report.append(f"  Prime Calculation: {pc['primes_per_sec']:.0f} primes/sec")

        if 'hash' in tests:
            report.append(f"\n🔐 Hash Throughput:")
            for algo, perf in tests['hash'].items():
                report.append(f"  {algo.upper()}: {perf['throughput_mbps']:.1f} MB/s")

        if 'threading' in tests:
            th = tests['threading']
            report.append(f"\n🧵 Parallelization:")
            report.append(f"  Thread Efficiency: {th['efficiency_threads']*100:.1f}%")
            report.append(f"  Process Efficiency: {th['efficiency_process']*100:.1f}%")

        # Performance rating
        report.append("\n⭐ Performance Rating:")
        if 'matrix_mult' in tests:
            gflops = tests['matrix_mult']['avg_gflops']
            if gflops > 100:
                rating = "Excellent"
            elif gflops > 50:
                rating = "Good"
            elif gflops > 20:
                rating = "Moderate"
            else:
                rating = "Limited"
            report.append(f"  Overall: {rating} ({gflops:.1f} GFLOPS)")

        return "\n".join(report)


def main():
    """Run CPU benchmark standalone"""
    benchmark = CPUBenchmark()
    results = benchmark.run_comprehensive_test()
    print(benchmark.generate_report(results))

    # Save JSON results
    with open('cpu_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n💾 Results saved to cpu_benchmark_results.json")


if __name__ == "__main__":
    main()
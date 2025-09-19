#!/usr/bin/env python3
"""
Storage I/O Performance Benchmark Module
Tests disk read/write performance with various patterns
"""

import os
import time
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import string

# Import configuration
try:
    from config import BENCHMARK_CONFIG
    STORAGE_CONFIG = BENCHMARK_CONFIG.get('storage', {})
except ImportError:
    # Fallback if config not available
    STORAGE_CONFIG = {}


class StorageBenchmark:
    """Storage I/O performance testing utilities"""

    def __init__(self):
        self.test_locations = self._identify_test_locations()
        self.results = {}

    def _identify_test_locations(self) -> Dict[str, Path]:
        """Identify available test locations"""
        locations = {}

        # Current working directory
        locations['cwd'] = Path.cwd()

        # System temp directory
        locations['temp'] = Path(tempfile.gettempdir())

        # Model cache directory
        cache_candidates = [
            Path.home() / '.cache' / 'huggingface',
            Path.home() / '.cache' / 'qwen3_fast_loader',
            Path.home() / '.cache'
        ]
        for cache_dir in cache_candidates:
            if cache_dir.exists() and os.access(cache_dir, os.W_OK):
                locations['cache'] = cache_dir
                break

        # Check available space for each location
        for name, path in list(locations.items()):
            try:
                stat = shutil.disk_usage(path)
                min_free = STORAGE_CONFIG.get('min_free_space_gb', 10)
                if stat.free < min_free * 1024**3:
                    print(f"⚠️ {name} has limited space: {stat.free / 1024**3:.1f}GB free")
            except:
                pass

        return locations

    def _detect_storage_type(self, path: Path) -> str:
        """Detect if storage is SSD or HDD"""
        # Linux-specific detection via /sys/block
        try:
            # Find the device for this path
            import subprocess
            result = subprocess.run(
                ['df', str(path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    device = lines[1].split()[0]
                    # Extract base device name (e.g., sda from /dev/sda1)
                    if device.startswith('/dev/'):
                        device_name = device[5:].rstrip('0123456789')

                        # Check rotational flag
                        rotational_file = f'/sys/block/{device_name}/queue/rotational'
                        if os.path.exists(rotational_file):
                            with open(rotational_file, 'r') as f:
                                is_rotational = f.read().strip() == '1'
                                return 'HDD' if is_rotational else 'SSD'
        except:
            pass

        return 'Unknown'

    def sequential_read_test(self, location: Path, size_mb: int = None,
                           block_size_kb: int = None) -> Dict[str, float]:
        if size_mb is None:
            size_mb = STORAGE_CONFIG.get('sequential_size_mb', 1024)
        if block_size_kb is None:
            block_size_kb = STORAGE_CONFIG.get('sequential_block_kb', 4096)
        """Test sequential read performance"""
        results = {
            'location': str(location),
            'size_mb': size_mb,
            'block_size_kb': block_size_kb,
            'storage_type': self._detect_storage_type(location)
        }

        # Create test file
        test_file = location / f'storage_bench_{os.getpid()}.tmp'

        try:
            # Write test data
            print(f"  Creating {size_mb}MB test file...", end='', flush=True)
            write_start = time.perf_counter()

            with open(test_file, 'wb') as f:
                chunk_size = block_size_kb * 1024
                chunks = (size_mb * 1024 * 1024) // chunk_size
                data = os.urandom(chunk_size)

                for _ in range(chunks):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())

            write_time = time.perf_counter() - write_start
            results['write_time'] = write_time
            results['write_mbps'] = size_mb / write_time if write_time > 0 else 0
            print(f" ✓ ({results['write_mbps']:.1f} MB/s write)")

            # Clear caches if possible (Linux)
            try:
                os.system('sync')
                if os.geteuid() == 0:  # If running as root
                    os.system('echo 3 > /proc/sys/vm/drop_caches')
                time.sleep(0.5)  # Brief pause
            except:
                pass

            # Sequential read test
            print(f"  Testing sequential read...", end='', flush=True)
            read_times = []

            for iteration in range(3):
                # Clear any Python-level caching
                if iteration > 0:
                    time.sleep(0.1)

                read_start = time.perf_counter()

                with open(test_file, 'rb') as f:
                    bytes_read = 0
                    while True:
                        chunk = f.read(block_size_kb * 1024)
                        if not chunk:
                            break
                        bytes_read += len(chunk)

                read_time = time.perf_counter() - read_start
                read_times.append(read_time)

            avg_read_time = sum(read_times) / len(read_times)
            results['read_time'] = avg_read_time
            results['read_mbps'] = size_mb / avg_read_time if avg_read_time > 0 else 0
            results['read_times'] = read_times
            print(f" ✓ ({results['read_mbps']:.1f} MB/s)")

        except Exception as e:
            results['error'] = str(e)
            print(f" ✗ Error: {e}")
        finally:
            # Clean up test file
            if test_file.exists():
                try:
                    test_file.unlink()
                except:
                    pass

        return results

    def random_read_test(self, location: Path, file_size_mb: int = None,
                        read_count: int = None, block_size_kb: int = None) -> Dict[str, float]:
        if file_size_mb is None:
            file_size_mb = STORAGE_CONFIG.get('random_file_size_mb', 100)
        if read_count is None:
            read_count = STORAGE_CONFIG.get('random_read_count', 1000)
        if block_size_kb is None:
            block_size_kb = STORAGE_CONFIG.get('random_block_kb', 4)
        """Test random read IOPS"""
        results = {
            'location': str(location),
            'file_size_mb': file_size_mb,
            'read_count': read_count,
            'block_size_kb': block_size_kb
        }

        test_file = location / f'iops_bench_{os.getpid()}.tmp'

        try:
            # Create test file with random data
            with open(test_file, 'wb') as f:
                f.write(os.urandom(file_size_mb * 1024 * 1024))
                f.flush()
                os.fsync(f.fileno())

            # Random read test
            file_size_bytes = file_size_mb * 1024 * 1024
            block_size_bytes = block_size_kb * 1024
            max_offset = file_size_bytes - block_size_bytes

            start = time.perf_counter()

            with open(test_file, 'rb') as f:
                for _ in range(read_count):
                    offset = random.randint(0, max_offset)
                    f.seek(offset)
                    _ = f.read(block_size_bytes)

            elapsed = time.perf_counter() - start

            results['time'] = elapsed
            results['iops'] = read_count / elapsed if elapsed > 0 else 0
            results['latency_ms'] = (elapsed / read_count) * 1000 if read_count > 0 else 0

        except Exception as e:
            results['error'] = str(e)
        finally:
            if test_file.exists():
                try:
                    test_file.unlink()
                except:
                    pass

        return results

    def model_shard_simulation(self, location: Path) -> Dict[str, float]:
        """Simulate reading model shards (4.5GB chunks)"""
        # Use configured sizes for testing
        shard_size_mb = STORAGE_CONFIG.get('shard_size_mb', 450)
        num_shards = STORAGE_CONFIG.get('num_shards', 3)

        results = {
            'location': str(location),
            'shard_size_mb': shard_size_mb,
            'num_shards': num_shards,
            'storage_type': self._detect_storage_type(location)
        }

        print(f"  Simulating {num_shards} x {shard_size_mb}MB shard reads...")

        shard_times = []
        total_start = time.perf_counter()

        for i in range(num_shards):
            # Use larger blocks for model loading simulation
            shard_result = self.sequential_read_test(
                location,
                size_mb=shard_size_mb,
                block_size_kb=8192
            )
            if 'read_mbps' in shard_result:
                shard_times.append(shard_result['read_time'])
                print(f"    Shard {i+1}: {shard_result['read_mbps']:.1f} MB/s")

        total_time = time.perf_counter() - total_start

        if shard_times:
            results['total_time'] = total_time
            results['avg_shard_time'] = sum(shard_times) / len(shard_times)
            results['total_mbps'] = (shard_size_mb * len(shard_times)) / total_time
            results['estimated_80b_load_time'] = (40000 / results['total_mbps']) if results['total_mbps'] > 0 else 0
            print(f"  Estimated 80B model load time: {results['estimated_80b_load_time']:.1f} seconds")

        return results

    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run complete storage benchmark suite"""
        print("\n💾 Storage Performance Testing")
        print("=" * 50)

        results = {
            'locations': {},
            'summary': {}
        }

        for loc_name, loc_path in self.test_locations.items():
            print(f"\n📁 Testing {loc_name}: {loc_path}")
            print(f"  Storage Type: {self._detect_storage_type(loc_path)}")

            location_results = {}

            # Sequential read test
            seq_result = self.sequential_read_test(loc_path)
            location_results['sequential'] = seq_result

            # Random read IOPS test
            print(f"  Testing random read IOPS...", end='', flush=True)
            iops_result = self.random_read_test(loc_path)
            location_results['random'] = iops_result
            if 'iops' in iops_result:
                print(f" ✓ ({iops_result['iops']:.0f} IOPS)")
            else:
                print(" ✗")

            results['locations'][loc_name] = location_results

        # Find best location for model loading
        best_location = None
        best_speed = 0
        for loc_name, loc_results in results['locations'].items():
            if 'sequential' in loc_results and 'read_mbps' in loc_results['sequential']:
                speed = loc_results['sequential']['read_mbps']
                if speed > best_speed:
                    best_speed = speed
                    best_location = loc_name

        results['summary']['best_location'] = best_location
        results['summary']['best_speed_mbps'] = best_speed

        # Model shard simulation on best location
        if best_location:
            print(f"\n🎯 Model Loading Simulation (using {best_location}):")
            shard_result = self.model_shard_simulation(self.test_locations[best_location])
            results['model_simulation'] = shard_result

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("\n" + "="*60)
        report.append("💾 STORAGE PERFORMANCE REPORT")
        report.append("="*60)

        # Location results
        for loc_name, loc_results in results.get('locations', {}).items():
            report.append(f"\n📁 {loc_name.upper()}:")

            if 'sequential' in loc_results and 'read_mbps' in loc_results['sequential']:
                seq = loc_results['sequential']
                report.append(f"  Type: {seq.get('storage_type', 'Unknown')}")
                report.append(f"  Sequential Read: {seq['read_mbps']:.1f} MB/s")
                report.append(f"  Sequential Write: {seq.get('write_mbps', 0):.1f} MB/s")

            if 'random' in loc_results and 'iops' in loc_results['random']:
                rand = loc_results['random']
                report.append(f"  Random Read IOPS: {rand['iops']:.0f}")
                report.append(f"  Latency: {rand['latency_ms']:.2f} ms")

        # Summary
        summary = results.get('summary', {})
        if summary.get('best_location'):
            report.append(f"\n🎯 Best Location: {summary['best_location']}")
            report.append(f"   Speed: {summary['best_speed_mbps']:.1f} MB/s")

        # Model loading estimate
        if 'model_simulation' in results and 'estimated_80b_load_time' in results['model_simulation']:
            estimate = results['model_simulation']['estimated_80b_load_time']
            report.append(f"\n⏱️ Estimated Model Load Time:")
            report.append(f"   80B Model (~40GB): {estimate:.0f} seconds ({estimate/60:.1f} minutes)")

        return "\n".join(report)


def main():
    """Run storage benchmark standalone"""
    benchmark = StorageBenchmark()
    results = benchmark.run_comprehensive_test()
    print(benchmark.generate_report(results))

    # Save JSON results
    with open('storage_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n💾 Results saved to storage_benchmark_results.json")


if __name__ == "__main__":
    main()
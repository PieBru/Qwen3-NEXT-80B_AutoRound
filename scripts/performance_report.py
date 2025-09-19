#!/usr/bin/env python3
"""
Unified Performance Report Generator
Combines results from all benchmark modules into comprehensive reports
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class PerformanceReportGenerator:
    """Generate comprehensive performance reports"""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.results = {}

    def add_results(self, category: str, data: Dict) -> None:
        """Add benchmark results for a category"""
        self.results[category] = data

    def generate_json_report(self) -> Dict:
        """Generate complete JSON report"""
        report = {
            'timestamp': self.timestamp,
            'system_info': self._get_system_info(),
            'benchmarks': self.results,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        return report

    def _get_system_info(self) -> Dict:
        """Gather system information"""
        import platform
        import psutil

        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        return info

    def _generate_summary(self) -> Dict:
        """Generate performance summary"""
        summary = {
            'performance_score': 0,
            'bottlenecks': [],
            'strengths': [],
            'overall_rating': 'Unknown'
        }

        scores = []

        # Memory performance scoring
        if 'memory' in self.results:
            mem = self.results['memory']
            ram_bw = mem.get('analysis', {}).get('ram_bandwidth', 0)
            if ram_bw > 0:
                # Score based on bandwidth (0-100)
                mem_score = min(100, (ram_bw / 100) * 100)  # 100GB/s = perfect score
                scores.append(mem_score)

                if ram_bw < 20:
                    summary['bottlenecks'].append('Low memory bandwidth')
                elif ram_bw > 50:
                    summary['strengths'].append('High memory bandwidth')

        # CPU performance scoring
        if 'cpu' in self.results:
            cpu = self.results['cpu']
            gflops = cpu.get('tests', {}).get('matrix_mult', {}).get('avg_gflops', 0)
            if gflops > 0:
                # Score based on GFLOPS (0-100)
                cpu_score = min(100, (gflops / 200) * 100)  # 200 GFLOPS = perfect score
                scores.append(cpu_score)

                if gflops < 20:
                    summary['bottlenecks'].append('Limited CPU compute')
                elif gflops > 100:
                    summary['strengths'].append('Strong CPU compute')

            # Check for SIMD support
            features = cpu.get('cpu_info', {}).get('features', [])
            if 'AVX2' in features or 'AVX512F' in features:
                summary['strengths'].append('Advanced SIMD support')

        # Storage performance scoring
        if 'storage' in self.results:
            storage = self.results['storage']
            read_speed = storage.get('summary', {}).get('best_speed_mbps', 0)
            if read_speed > 0:
                # Score based on read speed (0-100)
                storage_score = min(100, (read_speed / 5000) * 100)  # 5GB/s = perfect score
                scores.append(storage_score)

                if read_speed < 100:
                    summary['bottlenecks'].append('Slow storage I/O')
                elif read_speed > 1000:
                    summary['strengths'].append('Fast SSD storage')

        # Calculate overall score
        if scores:
            summary['performance_score'] = sum(scores) / len(scores)

            if summary['performance_score'] >= 80:
                summary['overall_rating'] = 'Excellent'
            elif summary['performance_score'] >= 60:
                summary['overall_rating'] = 'Good'
            elif summary['performance_score'] >= 40:
                summary['overall_rating'] = 'Moderate'
            else:
                summary['overall_rating'] = 'Limited'

        return summary

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        summary = self._generate_summary()

        # Memory recommendations
        if 'memory' in self.results:
            mem = self.results['memory']
            ram_bw = mem.get('analysis', {}).get('ram_bandwidth', 0)
            sys_mem = mem.get('system_memory_gb', 0)

            if sys_mem < 64:
                recommendations.append("Consider upgrading to at least 64GB RAM for optimal 80B model performance")
            if ram_bw < 20:
                recommendations.append("Memory bandwidth is low - consider faster RAM or different memory configuration")

        # CPU recommendations
        if 'cpu' in self.results:
            cpu = self.results['cpu']
            cores = cpu.get('cpu_info', {}).get('physical_cores', 0)
            gflops = cpu.get('tests', {}).get('matrix_mult', {}).get('avg_gflops', 0)

            if cores < 8:
                recommendations.append("More CPU cores would improve inference performance")
            if gflops < 50:
                recommendations.append("CPU compute performance is limited - newer CPU recommended")

        # Storage recommendations
        if 'storage' in self.results:
            storage = self.results['storage']
            read_speed = storage.get('summary', {}).get('best_speed_mbps', 0)
            storage_type = None
            for loc in storage.get('locations', {}).values():
                if 'sequential' in loc:
                    storage_type = loc['sequential'].get('storage_type')
                    break

            if storage_type == 'HDD':
                recommendations.append("SSD storage strongly recommended for faster model loading")
            elif read_speed < 500:
                recommendations.append("Consider NVMe SSD for faster model loading times")

        # Model loading strategy recommendations
        if summary['overall_rating'] == 'Excellent':
            recommendations.append("Use 'max-gpu' or 'min-gpu' mode for best performance")
        elif summary['overall_rating'] == 'Good':
            recommendations.append("Use 'min-gpu' mode for balanced performance")
        else:
            recommendations.append("Use 'no-gpu' mode with caching for best experience on this hardware")
            recommendations.append("First run will be slow, but cached runs will be much faster")

        # General recommendations
        if not summary['bottlenecks'] and summary['overall_rating'] in ['Excellent', 'Good']:
            recommendations.append("Your system is well-configured for running large language models")

        return recommendations

    def generate_console_report(self) -> str:
        """Generate formatted console report"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("         🎯 COMPREHENSIVE PERFORMANCE REPORT")
        lines.append("="*70)

        # Timestamp
        lines.append(f"Generated: {self.timestamp}")

        # System info
        sys_info = self._get_system_info()
        lines.append(f"\n📊 System Configuration:")
        lines.append(f"   Platform: {sys_info['platform']} {sys_info['architecture']}")
        lines.append(f"   CPUs: {sys_info['cpu_count']}P/{sys_info['cpu_count_logical']}L")
        lines.append(f"   Memory: {sys_info['memory_total_gb']:.1f}GB total, {sys_info['memory_available_gb']:.1f}GB available")

        # Performance summary
        summary = self._generate_summary()
        lines.append(f"\n⭐ Performance Score: {summary['performance_score']:.1f}/100 ({summary['overall_rating']})")

        if summary['strengths']:
            lines.append(f"\n✅ Strengths:")
            for strength in summary['strengths']:
                lines.append(f"   • {strength}")

        if summary['bottlenecks']:
            lines.append(f"\n⚠️ Bottlenecks:")
            for bottleneck in summary['bottlenecks']:
                lines.append(f"   • {bottleneck}")

        # Detailed results
        lines.append(f"\n📈 Benchmark Results:")

        if 'memory' in self.results:
            mem = self.results['memory']
            if mem.get('analysis', {}).get('ram_bandwidth'):
                lines.append(f"\n   Memory Performance:")
                lines.append(f"      RAM Bandwidth: {mem['analysis']['ram_bandwidth']:.1f} GB/s")
                if mem['analysis'].get('l3_bandwidth'):
                    lines.append(f"      L3 Cache: {mem['analysis']['l3_bandwidth']:.1f} GB/s")

        if 'cpu' in self.results:
            cpu = self.results['cpu']
            tests = cpu.get('tests', {})
            if 'matrix_mult' in tests:
                lines.append(f"\n   CPU Performance:")
                lines.append(f"      Compute: {tests['matrix_mult']['avg_gflops']:.1f} GFLOPS")
            if 'threading' in tests:
                lines.append(f"      Thread Efficiency: {tests['threading']['efficiency_threads']*100:.1f}%")

        if 'storage' in self.results:
            storage = self.results['storage']
            summary_data = storage.get('summary', {})
            if summary_data.get('best_speed_mbps'):
                lines.append(f"\n   Storage Performance:")
                lines.append(f"      Best Read Speed: {summary_data['best_speed_mbps']:.1f} MB/s")
            if 'model_simulation' in storage:
                est_time = storage['model_simulation'].get('estimated_80b_load_time', 0)
                if est_time > 0:
                    lines.append(f"      Est. Model Load: {est_time:.0f}s ({est_time/60:.1f}m)")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            lines.append(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"   {i}. {rec}")

        lines.append("\n" + "="*70)
        lines.append("          End of Performance Report")
        lines.append("="*70)

        return "\n".join(lines)

    def save_reports(self, base_path: str = "performance_report") -> tuple:
        """Save both JSON and text reports"""
        # Save JSON report
        json_path = f"{base_path}.json"
        with open(json_path, 'w') as f:
            json.dump(self.generate_json_report(), f, indent=2, default=str)

        # Save text report
        txt_path = f"{base_path}.txt"
        with open(txt_path, 'w') as f:
            f.write(self.generate_console_report())

        return json_path, txt_path


def combine_benchmark_results(memory_results: Optional[Dict] = None,
                             cpu_results: Optional[Dict] = None,
                             storage_results: Optional[Dict] = None) -> PerformanceReportGenerator:
    """Combine individual benchmark results into unified report"""
    generator = PerformanceReportGenerator()

    if memory_results:
        generator.add_results('memory', memory_results)
    if cpu_results:
        generator.add_results('cpu', cpu_results)
    if storage_results:
        generator.add_results('storage', storage_results)

    return generator


def main():
    """Generate report from saved benchmark files"""
    generator = PerformanceReportGenerator()

    # Try to load existing benchmark results
    try:
        with open('memory_benchmark_results.json', 'r') as f:
            generator.add_results('memory', json.load(f))
        print("✓ Loaded memory benchmark results")
    except FileNotFoundError:
        print("✗ No memory benchmark results found")

    try:
        with open('cpu_benchmark_results.json', 'r') as f:
            generator.add_results('cpu', json.load(f))
        print("✓ Loaded CPU benchmark results")
    except FileNotFoundError:
        print("✗ No CPU benchmark results found")

    try:
        with open('storage_benchmark_results.json', 'r') as f:
            generator.add_results('storage', json.load(f))
        print("✓ Loaded storage benchmark results")
    except FileNotFoundError:
        print("✗ No storage benchmark results found")

    # Generate and display report
    print(generator.generate_console_report())

    # Save reports
    json_path, txt_path = generator.save_reports()
    print(f"\n📄 Reports saved:")
    print(f"   JSON: {json_path}")
    print(f"   Text: {txt_path}")


if __name__ == "__main__":
    main()
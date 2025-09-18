# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-18-hardware-performance-cli/spec.md

## Technical Requirements

### Memory Performance Testing
- Implement memory bandwidth testing using numpy arrays with different sizes:
  - Small blocks: 1MB (L1/L2 cache performance)
  - Medium blocks: 100MB (L3 cache and RAM)
  - Large blocks: 1-10GB (pure RAM bandwidth)
- Use numpy.copy() and time.perf_counter() for accurate measurements
- Test both sequential and random access patterns
- Calculate throughput in GB/s

### CPU Performance Metrics
- Extract BogoMIPS from /proc/cpuinfo or dmesg output
- Implement simple CPU benchmark:
  - Matrix multiplication test (numpy)
  - Prime number calculation (pure Python)
  - Hash calculation benchmark (hashlib)
- Detect CPU features:
  - AVX/AVX2/AVX-512 support via cpuinfo
  - Core count and thread configuration
  - CPU frequency (base and boost)
- Multi-threaded vs single-threaded performance comparison

### Storage I/O Testing
- Sequential read test with 4.5GB blocks (matching model shard size)
- Use direct I/O when possible to bypass OS cache
- Test locations:
  - Model cache directory (~/.cache/huggingface/)
  - System temp directory
  - Current working directory
- Measure:
  - Sequential read speed (MB/s)
  - IOPS for small random reads
  - Latency for first byte

### System Information Gathering
- Parse /proc/cpuinfo for detailed CPU information
- Parse /proc/meminfo for memory configuration
- Use lscpu, dmidecode (if available) for additional details
- Detect storage type (SSD vs HDD) via:
  - /sys/block/*/queue/rotational
  - smartctl output analysis
- Gather thermal throttling information if available

### Performance Report Format
- Console output with formatted tables
- JSON output structure for automation
- Comparative analysis against reference systems
- Performance predictions for model operations:
  - Estimated model loading time
  - Expected tokens/second range
  - Memory bandwidth utilization percentage

### Integration Points
- Add new CLI arguments to existing argparse structure
- Integrate with existing benchmark_hardware.py functionality
- Store results in benchmark JSON format for consistency
- Hook into existing system info detection code

## External Dependencies (Conditional)

- **py-cpuinfo** - For cross-platform CPU feature detection
  - Justification: Provides reliable CPU feature detection across different systems
- **psutil** - Already in requirements, use for process/system metrics
  - Justification: Already available, extends current usage
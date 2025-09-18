#!/bin/bash

# Intel Optimizations Installation Script for AutoRound
# This script installs optional Intel optimizations that can improve performance

echo "========================================================"
echo "Intel Optimizations for AutoRound Installation"
echo "========================================================"
echo ""
echo "This will install optional optimizations that can improve"
echo "CPU inference performance for the Qwen3-80B model."
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first."
    exit 1
fi

echo "Available optimizations:"
echo ""
echo "1. Intel Extension for PyTorch (IPEX)"
echo "   - Optimizes PyTorch operations on Intel CPUs"
echo "   - Can provide 2-4x speedup for CPU inference"
echo ""
echo "2. Intel Neural Compressor (INC)"
echo "   - Additional quantization optimizations"
echo "   - Works with AutoRound for better performance"
echo ""

read -p "Install Intel Extension for PyTorch? (y/n): " install_ipex

if [ "$install_ipex" = "y" ]; then
    echo "Checking Python version compatibility..."
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    python_impl=$(python -c "import sys; print('free-threaded' if hasattr(sys, '_is_gil_enabled') else 'standard')")

    echo "  Python version: $python_version ($python_impl build)"

    if [[ "$python_version" == "3.13" ]] && [[ "$python_impl" == "free-threaded" ]]; then
        echo "⚠️  Intel Extension for PyTorch is not yet compatible with Python 3.13 free-threaded build"
        echo "   IPEX currently supports Python 3.9-3.12 with standard builds"
        echo "   Skipping IPEX installation..."
    else
        echo "Installing Intel Extension for PyTorch..."
        uv pip install intel-extension-for-pytorch
        if [ $? -eq 0 ]; then
            echo "✅ Intel Extension for PyTorch installed successfully"
        else
            echo "⚠️  Failed to install Intel Extension for PyTorch"
            echo "   This might be due to Python version compatibility"
        fi
    fi
fi

read -p "Install Intel Neural Compressor? (y/n): " install_inc

if [ "$install_inc" = "y" ]; then
    echo "Installing Intel Neural Compressor..."
    uv pip install neural-compressor
    if [ $? -eq 0 ]; then
        echo "✅ Intel Neural Compressor installed successfully"
    else
        echo "⚠️  Failed to install Intel Neural Compressor"
    fi
fi

echo ""
echo "========================================================"
echo "Testing optimizations..."
echo "========================================================"

python -c "
import sys
print('Checking installed optimizations...')
print('')

try:
    import intel_extension_for_pytorch as ipex
    print('✅ Intel Extension for PyTorch: ' + ipex.__version__)
except ImportError:
    print('❌ Intel Extension for PyTorch: Not installed')

try:
    import neural_compressor
    print('✅ Intel Neural Compressor: Installed')
except ImportError:
    print('❌ Intel Neural Compressor: Not installed')

try:
    import auto_round
    print('✅ AutoRound: ' + auto_round.__version__)
except ImportError:
    print('❌ AutoRound: Not installed')

print('')
print('CPU Information:')
import platform
print(f'  Processor: {platform.processor()}')
import os
print(f'  CPU Count: {os.cpu_count()} cores')
"

echo ""
echo "========================================================"
echo "Installation complete!"
echo ""
echo "To use the optimized script, run:"
echo "  python qwen3_80b_autoround_optimized.py"
echo ""
echo "For CPU-only inference with optimizations:"
echo "  python qwen3_80b_autoround_optimized.py --cpu"
echo "========================================================"
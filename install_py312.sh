#!/bin/bash

# Quick installation script for Python 3.12 with IPEX support
# This provides the best performance for CPU inference

set -e  # Exit on error
export UV_LINK_MODE=copy

echo "========================================================"
echo "Python 3.12 + IPEX Quick Setup"
echo "========================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" != "3.11" && "$python_version" != "3.12" ]]; then
    echo -e "${YELLOW}Warning: Python $python_version detected${NC}"
    echo "This script is optimized for Python 3.11 or 3.12"
    read -p "Continue anyway? (y/n): " continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing PyTorch and base requirements"
echo "================================================"

# Install base requirements first
if command -v uv &> /dev/null; then
    echo "Using uv for fast installation..."
    echo ""

    # Install PyTorch 2.5.1 with CUDA support (latest stable that works with our model)
    echo "Installing PyTorch 2.5.1 with CUDA support..."
    uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo ""
    echo "Installing base requirements..."
    uv pip install -r requirements-base.txt
else
    echo "Using pip for installation..."
    echo ""
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements-base.txt
fi

echo ""
echo -e "${GREEN}✅ Base requirements installed${NC}"

echo ""
echo "Step 2: Intel Extension for PyTorch Status"
echo "=============================================="

# Check PyTorch version and IPEX compatibility
echo "Checking IPEX compatibility..."
pytorch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "unknown")
echo "Current PyTorch version: $pytorch_version"

# IPEX 2.5.0 requires PyTorch 2.8.x, but PyTorch 2.8 is not yet stable
# and our model requires PyTorch 2.5.1 for compatibility
if [[ "$pytorch_version" == "2.5"* ]]; then
    echo ""
    echo -e "${YELLOW}⚠️  Intel Extension for PyTorch requires PyTorch 2.8.x${NC}"
    echo "   Current PyTorch $pytorch_version is required for model compatibility"
    echo "   IPEX will be available once PyTorch 2.8 becomes stable"
    echo ""
    echo "   Alternative optimizations will be used:"
    echo "   • Intel MKL for optimized math operations"
    echo "   • OpenMP for parallel processing"
    echo "   • AutoRound quantization for reduced memory usage"
    ipex_installed=false
elif [[ "$pytorch_version" == "2.8"* ]]; then
    echo "PyTorch 2.8.x detected - installing IPEX..."
    if command -v uv &> /dev/null; then
        if uv pip install intel-extension-for-pytorch==2.5.0; then
            echo -e "${GREEN}✅ Intel Extension for PyTorch installed successfully!${NC}"
            ipex_installed=true
        else
            echo -e "${RED}❌ IPEX installation failed${NC}"
            ipex_installed=false
        fi
    else
        if pip install intel-extension-for-pytorch==2.5.0; then
            echo -e "${GREEN}✅ Intel Extension for PyTorch installed successfully!${NC}"
            ipex_installed=true
        else
            echo -e "${RED}❌ IPEX installation failed${NC}"
            ipex_installed=false
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Unsupported PyTorch version for IPEX${NC}"
    ipex_installed=false
fi

echo ""
echo "Step 3: Installing additional optimizations"
echo "==========================================="

# Install MKL if not already installed
echo "Installing Intel MKL for optimized math operations..."
if command -v uv &> /dev/null; then
    uv pip install mkl mkl-service || echo "MKL installation skipped (may already be installed)"
else
    pip install mkl mkl-service || echo "MKL installation skipped (may already be installed)"
fi

echo ""
echo "Step 4: Verifying installation"
echo "=============================="

python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   CUDA: Not available")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import intel_extension_for_pytorch as ipex
    print(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
    print("   🚀 2-4x CPU speedup enabled!")
except ImportError:
    print("⚠️  Intel Extension for PyTorch: Not installed")
    print("   CPU inference will work but slower")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers: Not installed")

try:
    import auto_round
    print(f"✅ AutoRound: {auto_round.__version__}")
except ImportError:
    print("❌ AutoRound: Not installed")

# Check for Intel MKL
try:
    import torch
    if 'mkl' in torch.__config__.show().lower():
        print("✅ Intel MKL: Detected")
except:
    print("⚠️  Intel MKL: Not detected")
EOF

echo ""
echo "========================================================"
echo "Installation Complete!"
echo "========================================================"

if [ "$ipex_installed" = true ]; then
    echo -e "${GREEN}✅ IPEX is installed - you'll get 2-4x CPU speedup!${NC}"
    echo ""
    echo "To run with IPEX optimizations:"
    echo "  python qwen3_80b_ipex.py --interactive"
    echo ""
    echo "Or use the main script:"
    echo "  python qwen3_80b.py --load-strategy no-gpu  # CPU-only mode"
else
    echo -e "${YELLOW}⚠️  IPEX not installed - CPU inference will be slower${NC}"
    echo ""
    echo "To run the model:"
    echo "  python qwen3_80b.py --interactive"
fi

echo ""
echo "For best performance:"
echo "  • Use --load-strategy no-gpu for CPU-only inference"
echo "  • Monitor with htop during loading"
echo "  • Ensure 50GB+ RAM is available"
echo "========================================================"

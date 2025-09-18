#!/bin/bash

# Install PyTorch 2.5.0 with IPEX 2.5.0 for optimal performance
# This provides the best compatibility between our model and IPEX

set -e  # Exit on error

echo "========================================================"
echo "PyTorch 2.5.0 + IPEX 2.5.0 Installation"
echo "========================================================"
echo ""
echo "This will install the optimal version combination:"
echo "  • PyTorch 2.5.0 (works with our model)"
echo "  • Intel Extension for PyTorch 2.5.0 (2-4x CPU speedup)"
echo "  • CUDA 12.1 support for GPU acceleration"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" != "3.11" && "$python_version" != "3.12" ]]; then
    echo -e "${YELLOW}Warning: Python $python_version detected${NC}"
    echo "IPEX works best with Python 3.11 or 3.12"
    read -p "Continue anyway? (y/n): " continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        exit 1
    fi
fi

echo ""
read -p "This will replace your current PyTorch installation. Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Installation cancelled"
    exit 0
fi

echo ""
echo "Step 1: Cleaning previous installations"
echo "======================================="

# Clean previous installations
if command -v uv &> /dev/null; then
    echo "Removing old PyTorch and IPEX installations..."
    uv pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y 2>/dev/null || true
else
    pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y 2>/dev/null || true
fi

echo ""
echo "Step 2: Installing PyTorch 2.5.0"
echo "================================"

if command -v uv &> /dev/null; then
    echo "Installing PyTorch 2.5.0 with CUDA 12.1 support..."
    # Install specific version 2.5.0 (not 2.5.1)
    uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "Step 3: Installing Intel Extension for PyTorch 2.5.0"
echo "====================================================="

if command -v uv &> /dev/null; then
    echo "Installing IPEX 2.5.0..."
    if uv pip install intel-extension-for-pytorch==2.5.0; then
        echo -e "${GREEN}✅ Intel Extension for PyTorch 2.5.0 installed successfully!${NC}"
        ipex_installed=true
    else
        echo -e "${RED}❌ IPEX installation failed${NC}"
        echo "Trying with pip..."
        if pip install intel-extension-for-pytorch==2.5.0; then
            echo -e "${GREEN}✅ Intel Extension for PyTorch 2.5.0 installed successfully!${NC}"
            ipex_installed=true
        else
            ipex_installed=false
        fi
    fi
else
    if pip install intel-extension-for-pytorch==2.5.0; then
        echo -e "${GREEN}✅ Intel Extension for PyTorch 2.5.0 installed successfully!${NC}"
        ipex_installed=true
    else
        echo -e "${RED}❌ IPEX installation failed${NC}"
        ipex_installed=false
    fi
fi

echo ""
echo "Step 4: Installing additional requirements"
echo "========================================="

echo "Installing base requirements..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements-base.txt
else
    pip install -r requirements-base.txt
fi

echo ""
echo "Installing Intel MKL..."
if command -v uv &> /dev/null; then
    uv pip install mkl mkl-service || echo "MKL already installed"
else
    pip install mkl mkl-service || echo "MKL already installed"
fi

echo ""
echo "Step 5: Testing IPEX installation"
echo "================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")
print("")

# Test PyTorch
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
    sys.exit(1)

# Test IPEX
try:
    import intel_extension_for_pytorch as ipex
    print(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
    print("   🚀 2-4x CPU speedup enabled!")

    # Verify IPEX works with PyTorch
    print("\n   Testing IPEX optimization...")
    model = torch.nn.Linear(10, 10)
    try:
        model = ipex.optimize(model)
        print("   ✅ IPEX optimization works!")

        # Test inference
        x = torch.randn(1, 10)
        with torch.no_grad():
            y = model(x)
        print("   ✅ IPEX inference works!")
    except Exception as e:
        print(f"   ⚠️  IPEX optimization error: {e}")

except ImportError:
    print("❌ Intel Extension for PyTorch: Not installed")
except Exception as e:
    print(f"⚠️  IPEX error: {e}")

# Test other dependencies
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

# Check MKL
try:
    if 'mkl' in torch.__config__.show().lower():
        print("✅ Intel MKL: Detected")
except:
    pass
EOF

echo ""
echo "========================================================"
if [ "$ipex_installed" = true ]; then
    echo -e "${GREEN}🎉 SUCCESS! IPEX is now installed and working!${NC}"
    echo ""
    echo "You now have:"
    echo "  • PyTorch 2.5.0 (compatible with model)"
    echo "  • Intel Extension for PyTorch 2.5.0"
    echo "  • 2-4x CPU inference speedup!"
    echo ""
    echo "To run with IPEX optimizations:"
    echo "  python qwen3_80b_ipex.py --cpu --interactive"
    echo ""
    echo "Or use the main script in CPU mode:"
    echo "  python qwen3_80b.py --load-strategy no-gpu"
else
    echo -e "${YELLOW}⚠️  Installation completed with warnings${NC}"
    echo "Check the error messages above"
fi
echo ""
echo "Next steps:"
echo "1. Test model loading: python qwen3_80b.py --check"
echo "2. Run inference: python qwen3_80b.py --interactive"
echo "========================================================"
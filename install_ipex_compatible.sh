#!/bin/bash

# Install IPEX-compatible versions of PyTorch and Intel Extension
# This script installs older compatible versions for IPEX support

set -e  # Exit on error

echo "========================================================"
echo "IPEX-Compatible Installation"
echo "========================================================"
echo ""
echo "This will install:"
echo "  • PyTorch 2.4.0 (compatible with IPEX 2.4.0)"
echo "  • Intel Extension for PyTorch 2.4.0"
echo "  • All required dependencies"
echo ""
echo "Note: This uses an older PyTorch version for IPEX compatibility"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

read -p "Continue? This will replace current PyTorch installation (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Installation cancelled"
    exit 0
fi

echo ""
echo "Step 1: Uninstalling current PyTorch"
echo "===================================="

# Uninstall current PyTorch to avoid conflicts
if command -v uv &> /dev/null; then
    echo "Removing current PyTorch installation..."
    uv pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y 2>/dev/null || true
else
    pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y 2>/dev/null || true
fi

echo ""
echo "Step 2: Installing PyTorch 2.4.0 with CUDA 12.1"
echo "================================================"

if command -v uv &> /dev/null; then
    echo "Installing PyTorch 2.4.0..."
    uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "Step 3: Installing Intel Extension for PyTorch 2.4.0"
echo "====================================================="

if command -v uv &> /dev/null; then
    echo "Installing IPEX 2.4.0..."
    if uv pip install intel-extension-for-pytorch==2.4.0; then
        echo -e "${GREEN}✅ Intel Extension for PyTorch installed successfully!${NC}"
        ipex_installed=true
    else
        echo -e "${RED}❌ IPEX installation failed${NC}"
        ipex_installed=false
    fi
else
    if pip install intel-extension-for-pytorch==2.4.0; then
        echo -e "${GREEN}✅ Intel Extension for PyTorch installed successfully!${NC}"
        ipex_installed=true
    else
        echo -e "${RED}❌ IPEX installation failed${NC}"
        ipex_installed=false
    fi
fi

echo ""
echo "Step 4: Installing other requirements"
echo "====================================="

echo "Installing base requirements..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements-base.txt
else
    pip install -r requirements-base.txt
fi

echo ""
echo "Step 5: Verifying installation"
echo "=============================="

python << EOF
import sys
print(f"Python: {sys.version}")
print("")

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

    # Test IPEX optimization
    print("\nTesting IPEX optimization...")
    import torch
    model = torch.nn.Linear(10, 10)
    model = ipex.optimize(model)
    print("   ✅ IPEX optimization works!")
except ImportError:
    print("❌ Intel Extension for PyTorch: Not installed")
except Exception as e:
    print(f"⚠️  IPEX installed but optimization test failed: {e}")

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
EOF

echo ""
echo "========================================================"
if [ "$ipex_installed" = true ]; then
    echo -e "${GREEN}✅ IPEX Successfully Installed!${NC}"
    echo ""
    echo "You now have Intel Extension for PyTorch for 2-4x CPU speedup!"
    echo ""
    echo "To use IPEX optimizations:"
    echo "  python qwen3_80b_ipex.py --interactive"
    echo ""
    echo "Or with the main script:"
    echo "  python qwen3_80b.py --load-strategy no-gpu"
else
    echo -e "${YELLOW}⚠️  Installation completed with warnings${NC}"
fi
echo ""
echo "Note: Using PyTorch 2.4.0 for IPEX compatibility"
echo "If you encounter model loading issues, you may need to"
echo "upgrade back to PyTorch 2.5.1 with ./install_py312.sh"
echo "========================================================"
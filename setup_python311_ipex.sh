#!/bin/bash

# Setup Python 3.11 environment with Intel Extension for PyTorch
# This provides 2-4x speedup for CPU inference with IPEX optimizations

set -e  # Exit on error

echo "========================================================"
echo "Python 3.11 + Intel Extension for PyTorch Setup"
echo "========================================================"
echo ""
echo "This script will:"
echo "1. Install Python 3.11 using pyenv"
echo "2. Create a virtual environment with uv"
echo "3. Install all dependencies including IPEX"
echo "4. Configure for optimal performance"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${YELLOW}pyenv not found. Installing pyenv...${NC}"
    curl https://pyenv.run | bash

    # Add to shell
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    # Load for current session
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    echo -e "${GREEN}✅ pyenv installed${NC}"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ""
echo "Step 1: Installing Python 3.11.10"
echo "=================================="

# Check if Python 3.11.10 is already installed
if pyenv versions | grep -q "3.11.10"; then
    echo -e "${GREEN}✅ Python 3.11.10 already installed${NC}"
else
    echo "Installing Python 3.11.10 (this may take a few minutes)..."
    pyenv install 3.11.10
    echo -e "${GREEN}✅ Python 3.11.10 installed${NC}"
fi

echo ""
echo "Step 2: Creating Virtual Environment"
echo "====================================="

# Set local Python version
pyenv local 3.11.10

# Check if venv already exists
if [ -d ".venv-py311-ipex" ]; then
    echo -e "${YELLOW}Virtual environment .venv-py311-ipex already exists${NC}"
    read -p "Delete and recreate? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        rm -rf .venv-py311-ipex
        uv venv .venv-py311-ipex --python=python3.11
    fi
else
    uv venv .venv-py311-ipex --python=python3.11
fi

echo -e "${GREEN}✅ Virtual environment created${NC}"

echo ""
echo "Step 3: Activating Environment"
echo "=============================="

source .venv-py311-ipex/bin/activate

python_version=$(python --version)
echo "Active Python: $python_version"

echo ""
echo "Step 4: Installing Core Dependencies"
echo "===================================="

echo "Installing PyTorch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Transformers from source (for latest Qwen3 support)..."
uv pip install --upgrade git+https://github.com/huggingface/transformers.git

echo "Installing other dependencies..."
uv pip install \
    accelerate \
    sentencepiece \
    protobuf \
    auto-round \
    psutil \
    huggingface-hub \
    safetensors \
    tqdm \
    numpy \
    scipy

echo -e "${GREEN}✅ Core dependencies installed${NC}"

echo ""
echo "Step 5: Installing Intel Extension for PyTorch"
echo "=============================================="

echo "Installing IPEX (this provides 2-4x CPU speedup)..."
uv pip install intel-extension-for-pytorch

if python -c "import intel_extension_for_pytorch as ipex; print(f'IPEX version: {ipex.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✅ Intel Extension for PyTorch installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  IPEX installation may have issues, continuing...${NC}"
fi

echo ""
echo "Step 6: Installing Optional Optimizations"
echo "========================================="

read -p "Install Intel Neural Compressor for additional optimizations? (y/n): " install_inc
if [ "$install_inc" = "y" ]; then
    uv pip install neural-compressor
    echo -e "${GREEN}✅ Intel Neural Compressor installed${NC}"
fi

read -p "Install Intel OpenVINO for inference optimization? (y/n): " install_ov
if [ "$install_ov" = "y" ]; then
    uv pip install openvino optimum-intel
    echo -e "${GREEN}✅ OpenVINO installed${NC}"
fi

echo ""
echo "Step 7: Testing Installation"
echo "============================"

python << EOF
import sys
print(f"Python version: {sys.version}")
print("")

# Test imports
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import intel_extension_for_pytorch as ipex
    print(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
    print("   CPU optimizations available!")
except ImportError:
    print("❌ Intel Extension for PyTorch: Not installed")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import auto_round
    print(f"✅ AutoRound: {auto_round.__version__}")
except ImportError:
    print("❌ AutoRound: Not installed")

try:
    import neural_compressor
    print("✅ Intel Neural Compressor: Installed")
except ImportError:
    print("ℹ️  Intel Neural Compressor: Not installed (optional)")

try:
    import openvino
    print(f"✅ OpenVINO: {openvino.__version__}")
except ImportError:
    print("ℹ️  OpenVINO: Not installed (optional)")

# Check CPU features
print("")
print("CPU Optimization Features:")
import platform
print(f"  Processor: {platform.processor()}")
print(f"  CPU count: {torch.get_num_threads()} threads")

# Check for AVX support (important for performance)
import subprocess
try:
    result = subprocess.run(['lscpu'], capture_output=True, text=True)
    if 'avx' in result.stdout.lower():
        print("  ✅ AVX instructions supported")
    if 'avx2' in result.stdout.lower():
        print("  ✅ AVX2 instructions supported")
    if 'avx512' in result.stdout.lower():
        print("  ✅ AVX-512 instructions supported")
except:
    pass
EOF

echo ""
echo "========================================================"
echo "✅ Setup Complete!"
echo "========================================================"
echo ""
echo "To activate this environment in the future:"
echo "  source .venv-py311-ipex/bin/activate"
echo ""
echo "To run the model with IPEX optimizations:"
echo "  python qwen3_80b_ipex.py --interactive"
echo ""
echo "Performance tips:"
echo "  • IPEX provides 2-4x speedup for CPU inference"
echo "  • Use --cpu flag to force CPU-only mode"
echo "  • Monitor with htop during model loading"
echo ""
echo "Note: First run will still take time to load the model,"
echo "but inference will be significantly faster with IPEX!"
echo "========================================================"
# Python Version Setup Guide

## Using Specific Python Versions with uv

### Method 1: Specify Python Version Directly
```bash
# Create venv with Python 3.12
uv venv --python 3.12

# Or with explicit path
uv venv --python python3.12

# Or with full path if you have multiple installations
uv venv --python /usr/bin/python3.12
```

### Method 2: Using pyenv with uv
```bash
# Install Python 3.12 with pyenv
pyenv install 3.12.8
pyenv local 3.12.8

# Now uv will use the pyenv-selected Python
uv venv  # Will use Python 3.12.8

# Or explicitly
uv venv --python $(pyenv which python)
```

### Method 3: Create Named Environments
```bash
# Create separate environments for different Python versions
uv venv .venv-py312 --python 3.12
uv venv .venv-py311 --python 3.11
uv venv .venv-py313 --python 3.13

# Activate the one you want
source .venv-py312/bin/activate  # For Python 3.12 with IPEX support
```

## Complete Setup for Python 3.12 with IPEX

### Quick Setup (Recommended)
```bash
# 1. Install Python 3.12 if not already installed
pyenv install 3.12.8
pyenv local 3.12.8

# 2. Create virtual environment with Python 3.12
uv venv --python 3.12

# 3. Activate it
source .venv/bin/activate

# 4. Verify Python version
python --version  # Should show Python 3.12.x

# 5. Install requirements with IPEX
pip install -r setup/requirements-py311-312.txt
```

### Alternative: Use the Setup Script
```bash
# This script handles everything for you
./setup/setup_python311_ipex.sh

# It will:
# - Install Python 3.11 or use existing
# - Create .venv-py311-ipex environment
# - Install all dependencies including IPEX
# - Verify the installation
```

## Checking Current Python Version

```bash
# Check system default
python3 --version

# Check what uv will use by default
uv venv --python python --dry-run

# Check available Python versions
ls -la /usr/bin/python*  # System installations
pyenv versions           # pyenv-managed versions

# Check current pyenv setting
pyenv version
```

## Managing Multiple Python Versions

### Using pyenv (Recommended)
```bash
# Install multiple versions
pyenv install 3.11.10
pyenv install 3.12.8
pyenv install 3.13.1

# Set project-specific version
pyenv local 3.12.8  # Creates .python-version file

# Set global default
pyenv global 3.12.8

# Temporarily use a version
pyenv shell 3.12.8
```

### Environment Variables
```bash
# Force uv to use specific Python
UV_PYTHON=/usr/bin/python3.12 uv venv

# Or set it permanently for the session
export UV_PYTHON=python3.12
uv venv
```

## Troubleshooting

### "Python 3.12 not found"
```bash
# Check if Python 3.12 is installed
which python3.12

# If not found, install it:
# On Ubuntu/Debian
sudo apt install python3.12 python3.12-venv

# On Arch Linux
sudo pacman -S python312

# Or use pyenv (works everywhere)
pyenv install 3.12.8
```

### uv Still Using Wrong Python
```bash
# Clear any existing venv
rm -rf .venv

# Be explicit about Python version
uv venv --python python3.12

# Or use full path
uv venv --python /home/$(whoami)/.pyenv/versions/3.12.8/bin/python
```

## Quick Commands Reference

```bash
# Create Python 3.12 venv with uv
uv venv --python 3.12

# Create Python 3.11 venv (also has IPEX support)
uv venv --python 3.11

# Install IPEX-enabled requirements (for Python 3.11/3.12)
uv pip install -r setup/requirements-py311-312.txt

# Run with IPEX optimizations
python qwen3_80b.py --interactive
```

## Why Python 3.12 for This Project?

1. **IPEX Support**: Intel Extension for PyTorch works with Python 3.11-3.12
2. **2-4x CPU Speedup**: Significant performance improvement for CPU inference
3. **Stable**: Python 3.12 is stable and well-supported
4. **Compatible**: All dependencies work well with Python 3.12

## Summary

For best performance with the Qwen3-80B model:
1. Use Python 3.12 (or 3.11)
2. Install IPEX with `setup/requirements-py311-312.txt`
3. Run with `qwen3_80b.py` for optimized inference

The key command you need:
```bash
uv venv --python 3.12
```
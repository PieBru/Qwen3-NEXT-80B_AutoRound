#!/usr/bin/env python3
"""
Intelligent requirements installer for Qwen3-80B AutoRound
Automatically detects Python version and installs appropriate dependencies
"""

import sys
import subprocess
import os
from pathlib import Path

def get_python_version():
    """Get Python version info"""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    # Check if it's a free-threaded build
    is_freethreaded = hasattr(sys, '_is_gil_enabled')

    return {
        'major': version_info.major,
        'minor': version_info.minor,
        'micro': version_info.micro,
        'version_str': version_str,
        'is_freethreaded': is_freethreaded,
        'full_version': sys.version
    }

def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass
    return False

def get_requirements_file(python_info):
    """Determine which requirements file to use"""
    major = python_info['major']
    minor = python_info['minor']

    if major == 3:
        if minor in [11, 12]:
            return 'requirements-py311-312.txt', True  # True = IPEX available
        elif minor >= 13:
            return 'requirements-py313.txt', False  # False = IPEX not available
        elif minor == 10:
            print("⚠️  Python 3.10 detected - limited support")
            print("   Consider upgrading to Python 3.11 or 3.12 for best performance")
            return 'requirements-py311-312.txt', False  # Try 3.11 requirements
        else:
            print(f"❌ Python 3.{minor} is not supported")
            print("   Please use Python 3.11, 3.12, or 3.13+")
            sys.exit(1)
    else:
        print(f"❌ Python {major}.{minor} is not supported")
        print("   Please use Python 3.11, 3.12, or 3.13+")
        sys.exit(1)

def install_requirements(req_file, python_info, ipex_available):
    """Install requirements using pip or uv"""
    print(f"\n📦 Installing from {req_file}")

    # Check if uv is available
    use_uv = subprocess.run(['which', 'uv'], capture_output=True).returncode == 0

    if use_uv:
        installer = ['uv', 'pip', 'install']
        print("   Using uv for fast installation")
    else:
        installer = [sys.executable, '-m', 'pip', 'install']
        print("   Using pip for installation")

    # Add CUDA index URL if CUDA is available
    cuda_available = check_cuda()
    if cuda_available:
        print("   ✅ CUDA detected - will install GPU-enabled PyTorch")
        if python_info['minor'] >= 13:
            # For Python 3.13, use nightly builds
            installer.extend(['--index-url', 'https://download.pytorch.org/whl/nightly/cu124'])
        else:
            # For Python 3.11-3.12, use stable CUDA 12.1
            installer.extend(['--index-url', 'https://download.pytorch.org/whl/cu121'])
    else:
        print("   ℹ️  No CUDA detected - will install CPU-only PyTorch")
        installer.extend(['--index-url', 'https://download.pytorch.org/whl/cpu'])

    # Install requirements
    cmd = installer + ['-r', req_file]

    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("   This may take several minutes...\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Requirements installed successfully!")

        if ipex_available and python_info['minor'] in [11, 12]:
            print("\n🎉 Intel Extension for PyTorch (IPEX) included!")
            print("   You'll get 2-4x CPU performance boost")
        elif python_info['minor'] >= 13:
            print("\n⚠️  Note: IPEX is not yet compatible with Python 3.13")
            print("   CPU inference will be slower than Python 3.11/3.12")
            print("   Consider using Python 3.11 or 3.12 for better performance")
    else:
        print("\n❌ Installation failed")
        print("   Try running manually:")
        print(f"   {' '.join(cmd)}")
        sys.exit(1)

def verify_installation():
    """Verify key packages are installed"""
    print("\n🔍 Verifying installation...")

    packages_to_check = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('auto_round', 'AutoRound'),
        ('accelerate', 'Accelerate'),
    ]

    # Check IPEX only for Python 3.11/3.12
    python_info = get_python_version()
    if python_info['minor'] in [11, 12]:
        packages_to_check.append(('intel_extension_for_pytorch', 'Intel Extension for PyTorch'))

    all_good = True
    for module_name, display_name in packages_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {display_name}: {version}")

            # Special check for CUDA
            if module_name == 'torch':
                import torch
                if torch.cuda.is_available():
                    print(f"      CUDA: {torch.version.cuda}")
                    print(f"      GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("      CUDA: Not available (CPU-only)")

        except ImportError:
            print(f"   ❌ {display_name}: Not installed")
            all_good = False

    return all_good

def main():
    print("="*60)
    print("Qwen3-80B AutoRound Requirements Installer")
    print("="*60)

    # Get Python version info
    python_info = get_python_version()
    print(f"\n📊 System Information:")
    print(f"   Python version: {python_info['version_str']}")
    print(f"   Full version: {python_info['full_version'][:50]}...")
    if python_info['is_freethreaded']:
        print("   ✅ Free-threaded build (GIL-free)")

    # Check CUDA
    cuda_available = check_cuda()
    if cuda_available:
        print("   ✅ CUDA/GPU detected")
    else:
        print("   ℹ️  No CUDA/GPU detected")

    # Determine requirements file
    req_file, ipex_available = get_requirements_file(python_info)

    print(f"\n📋 Selected requirements: {req_file}")
    if ipex_available:
        print("   ✅ Will include Intel Extension for PyTorch (2-4x CPU speedup)")
    else:
        print("   ℹ️  IPEX not available for this Python version")

    # Ask for confirmation
    print("\n" + "="*60)
    response = input("Proceed with installation? (y/n): ").strip().lower()

    if response != 'y':
        print("Installation cancelled")
        sys.exit(0)

    # Install requirements
    install_requirements(req_file, python_info, ipex_available)

    # Verify installation
    if verify_installation():
        print("\n" + "="*60)
        print("🎉 Installation complete!")
        print("\nNext steps:")
        print("1. Test the installation:")
        print("   python qwen3_80b.py --check")
        print("\n2. Run the model:")
        if ipex_available and python_info['minor'] in [11, 12]:
            print("   python qwen3_80b_ipex.py --interactive  # With IPEX optimizations")
        else:
            print("   python qwen3_80b.py --interactive")
        print("\n3. For CPU-only mode:")
        print("   python qwen3_80b.py --load-strategy no-gpu")
        print("="*60)
    else:
        print("\n⚠️  Some packages may not have installed correctly")
        print("Please check the errors above and try again")

if __name__ == "__main__":
    main()
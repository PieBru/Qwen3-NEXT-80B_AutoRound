#!/bin/bash
# Wrapper script to ensure multi-core loading

echo "🔧 Setting up multi-core environment for 32 threads..."

# Set all threading environment variables
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export VECLIB_MAXIMUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export BLIS_NUM_THREADS=32

# Python GIL-free mode
export PYTHON_GIL=0

# GPU memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Force offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Environment configured:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  PYTHON_GIL=$PYTHON_GIL"
echo ""
echo "Starting model loading with multi-core support..."
echo "Monitor with: htop (should show all cores active)"
echo ""

# Run the optimized script
python run_qwen3_80b_optimized.py
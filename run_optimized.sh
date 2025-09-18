#!/bin/bash

# Optimized script for running Qwen3-80B with fast loading and IPEX inference
# This demonstrates the best workflow for CPU-only mode with IPEX

echo "========================================================"
echo "Optimized Qwen3-80B Launcher"
echo "========================================================"
echo ""
echo "This script ensures:"
echo "• Fast loading (IPEX disabled during load)"
echo "• Fast inference (IPEX enabled after load)"
echo "• 2-4x CPU speedup for generation"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Disable IPEX during loading phase
export IPEX_DISABLE_AUTO_OPTIMIZATION=1
export DISABLE_IPEX_AUTOCAST=1
export UV_LINK_MODE=copy

# Optional: Force single-threaded loading for consistency
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

echo -e "${YELLOW}Phase 1: Loading model (IPEX disabled for speed)${NC}"
echo "This prevents IPEX from slowing down the loading process"
echo ""

# Step 2: Run the model
# The script will automatically apply IPEX AFTER loading
echo -e "${GREEN}Starting model with optimized workflow...${NC}"
echo ""

# Run with CPU-only mode
# IPEX will be automatically applied after loading completes
python qwen3_80b.py --load-strategy no-gpu --interactive "$@"

echo ""
echo "========================================================"
echo "Session ended"
echo "========================================================"
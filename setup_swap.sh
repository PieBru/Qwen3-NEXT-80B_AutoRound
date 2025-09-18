#!/bin/bash
#
# Automatic swap setup for Qwen3-80B IPEX optimization
# For consumer hardware with 64-128GB RAM
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Qwen3-80B Swap Setup for IPEX Optimization${NC}"
echo -e "${BLUE}============================================${NC}"
echo

# Function to convert GB to bytes
gb_to_bytes() {
    echo $(($1 * 1024 * 1024 * 1024))
}

# Function to convert bytes to GB
bytes_to_gb() {
    echo $(($1 / 1024 / 1024 / 1024))
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo -e "${RED}Please run this script as a normal user (with sudo privileges)${NC}"
   echo "The script will request sudo when needed."
   exit 1
fi

# Get system information
TOTAL_RAM=$(free -b | grep '^Mem:' | awk '{print $2}')
AVAILABLE_RAM=$(free -b | grep '^Mem:' | awk '{print $7}')
CURRENT_SWAP=$(free -b | grep '^Swap:' | awk '{print $2}')

TOTAL_RAM_GB=$(bytes_to_gb $TOTAL_RAM)
AVAILABLE_RAM_GB=$(bytes_to_gb $AVAILABLE_RAM)
CURRENT_SWAP_GB=$(bytes_to_gb $CURRENT_SWAP)

echo -e "${GREEN}System Information:${NC}"
echo "  Total RAM: ${TOTAL_RAM_GB}GB"
echo "  Available RAM: ${AVAILABLE_RAM_GB}GB"
echo "  Current Swap: ${CURRENT_SWAP_GB}GB"
echo

# Calculate required swap for IPEX
# IPEX needs ~80GB peak during optimization
# Model runs at ~45GB after optimization
IPEX_PEAK_REQUIREMENT=80
RUNTIME_REQUIREMENT=45

TOTAL_CURRENT=$((TOTAL_RAM_GB + CURRENT_SWAP_GB))
NEEDED_FOR_IPEX=$((IPEX_PEAK_REQUIREMENT - TOTAL_CURRENT))

if [ $NEEDED_FOR_IPEX -le 0 ]; then
    echo -e "${GREEN}✓ You already have enough memory for IPEX optimization!${NC}"
    echo "  Total available: ${TOTAL_CURRENT}GB (RAM + Swap)"
    echo "  Required: ${IPEX_PEAK_REQUIREMENT}GB"
    echo
    read -p "Do you want to add more swap anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    NEEDED_FOR_IPEX=30  # Add 30GB extra if user wants more
fi

# Round up to nearest 10GB for safety
SWAP_TO_ADD=$(( (NEEDED_FOR_IPEX + 9) / 10 * 10 ))

echo -e "${YELLOW}Swap Recommendation:${NC}"
echo "  Need for IPEX: ${IPEX_PEAK_REQUIREMENT}GB total"
echo "  Currently have: ${TOTAL_CURRENT}GB (RAM + existing swap)"
echo "  Recommended to add: ${SWAP_TO_ADD}GB swap"
echo
echo -e "${BLUE}This swap is only needed for the ONE-TIME IPEX optimization.${NC}"
echo -e "${BLUE}After the model is cached, you can remove this swap.${NC}"
echo

# Check available disk space
AVAILABLE_DISK=$(df / | tail -1 | awk '{print $4}')
AVAILABLE_DISK_GB=$((AVAILABLE_DISK / 1024 / 1024))

if [ $AVAILABLE_DISK_GB -lt $SWAP_TO_ADD ]; then
    echo -e "${RED}ERROR: Not enough disk space!${NC}"
    echo "  Need: ${SWAP_TO_ADD}GB"
    echo "  Available: ${AVAILABLE_DISK_GB}GB"
    exit 1
fi

echo "Available disk space: ${AVAILABLE_DISK_GB}GB"
echo

# Ask for confirmation
read -p "Create ${SWAP_TO_ADD}GB swap file? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

# Create swap file
SWAPFILE="/swapfile_qwen3"
COUNTER=1
while [ -f "${SWAPFILE}" ]; do
    SWAPFILE="/swapfile_qwen3_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

echo
echo -e "${GREEN}Creating swap file: ${SWAPFILE}${NC}"
echo "This may take a few minutes..."

# Create the swap file
sudo fallocate -l ${SWAP_TO_ADD}G ${SWAPFILE} || {
    echo "fallocate failed, trying dd method (slower)..."
    sudo dd if=/dev/zero of=${SWAPFILE} bs=1G count=${SWAP_TO_ADD} status=progress
}

# Set permissions
sudo chmod 600 ${SWAPFILE}

# Make it a swap file
sudo mkswap ${SWAPFILE}

# Enable the swap
sudo swapon ${SWAPFILE}

# Verify
NEW_SWAP=$(free -b | grep '^Swap:' | awk '{print $2}')
NEW_SWAP_GB=$(bytes_to_gb $NEW_SWAP)

echo
echo -e "${GREEN}✓ Swap setup complete!${NC}"
echo "  Total RAM: ${TOTAL_RAM_GB}GB"
echo "  Total Swap: ${NEW_SWAP_GB}GB"
echo "  Total Available: $((TOTAL_RAM_GB + NEW_SWAP_GB))GB"
echo

# Create removal script
REMOVE_SCRIPT="remove_swap.sh"
cat > $REMOVE_SCRIPT << EOF
#!/bin/bash
# Remove temporary swap file created for IPEX optimization

echo "Removing swap file: ${SWAPFILE}"
sudo swapoff ${SWAPFILE}
sudo rm ${SWAPFILE}
echo "Swap file removed successfully"
free -h
EOF
chmod +x $REMOVE_SCRIPT

echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Run the model with IPEX optimization:"
echo "   python qwen3_80b.py --load-strategy no-gpu --interactive"
echo
echo "2. After the IPEX cache is created (first run), you can remove the swap:"
echo "   ./${REMOVE_SCRIPT}"
echo
echo -e "${GREEN}The swap file '${SWAPFILE}' will be automatically removed on reboot.${NC}"
echo -e "${GREEN}To make it permanent, add to /etc/fstab (not recommended for temporary use).${NC}"
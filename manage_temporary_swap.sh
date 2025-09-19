#!/bin/bash

# Manage temporary swap for Qwen3-80B model operations
# This script can create, check, or remove temporary swap space

set -e  # Exit on error

# Configuration
SWAP_FILE="/swapfile_temp"
MIN_TOTAL_MEMORY=160  # GB needed for IPEX optimization
RECOMMENDED_TOTAL=200  # GB recommended for comfortable operation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_color "$YELLOW" "⚠️  This script needs sudo privileges."
        echo "   Rerunning with sudo..."
        exec sudo "$0" "$@"
    fi
}

# Function to get memory info in GB
get_memory_info() {
    local total_ram=$(free -g | grep '^Mem:' | awk '{print $2}')
    local available_ram=$(free -g | grep '^Mem:' | awk '{print $7}')
    local total_swap=$(free -g | grep '^Swap:' | awk '{print $2}')
    local used_swap=$(free -g | grep '^Swap:' | awk '{print $3}')

    echo "$total_ram $available_ram $total_swap $used_swap"
}

# Function to display current memory status
show_status() {
    echo ""
    print_color "$BLUE" "📊 Current System Memory Status:"
    echo "================================"

    read total_ram available_ram total_swap used_swap <<< $(get_memory_info)
    local total_memory=$((total_ram + total_swap))

    echo "RAM:  ${total_ram}GB total, ${available_ram}GB available"
    echo "Swap: ${total_swap}GB total, ${used_swap}GB used"
    echo "--------------------------------"
    echo "Total: ${total_memory}GB (RAM + Swap)"
    echo ""

    # Check if temporary swap exists
    if [ -f "$SWAP_FILE" ]; then
        local swap_size=$(du -h "$SWAP_FILE" | cut -f1)
        if swapon --show | grep -q "$SWAP_FILE"; then
            print_color "$GREEN" "✅ Temporary swap is ACTIVE at $SWAP_FILE ($swap_size)"
        else
            print_color "$YELLOW" "⚠️  Temporary swap file exists but is INACTIVE at $SWAP_FILE ($swap_size)"
        fi
    else
        echo "❌ No temporary swap file at $SWAP_FILE"
    fi

    echo ""

    # Recommendations
    if [ $total_memory -lt $MIN_TOTAL_MEMORY ]; then
        print_color "$RED" "⚠️  WARNING: Only ${total_memory}GB total memory available"
        echo "   Minimum ${MIN_TOTAL_MEMORY}GB required for IPEX optimization"
        echo "   Need to add $((MIN_TOTAL_MEMORY - total_memory))GB more swap"
    elif [ $total_memory -lt $RECOMMENDED_TOTAL ]; then
        print_color "$YELLOW" "💡 TIP: You have ${total_memory}GB total memory"
        echo "   Recommended ${RECOMMENDED_TOTAL}GB for comfortable operation"
        echo "   Consider adding $((RECOMMENDED_TOTAL - total_memory))GB more swap"
    else
        print_color "$GREEN" "✅ Memory looks good! ${total_memory}GB should be sufficient"
    fi
}

# Function to create temporary swap
create_swap() {
    echo ""
    print_color "$BLUE" "🔧 Creating Temporary Swap Space"
    echo "================================="

    # Check if swap already exists
    if [ -f "$SWAP_FILE" ]; then
        if swapon --show | grep -q "$SWAP_FILE"; then
            print_color "$YELLOW" "⚠️  Temporary swap already active at $SWAP_FILE"
            show_status
            return 0
        else
            print_color "$YELLOW" "⚠️  Swap file exists but is not active. Activating..."
            swapon "$SWAP_FILE"
            print_color "$GREEN" "✅ Existing swap activated!"
            show_status
            return 0
        fi
    fi

    # Calculate how much swap to add
    read total_ram available_ram total_swap used_swap <<< $(get_memory_info)
    local total_memory=$((total_ram + total_swap))

    if [ $total_memory -ge $RECOMMENDED_TOTAL ]; then
        print_color "$GREEN" "✅ You already have ${total_memory}GB total memory!"
        echo "   No additional swap needed."
        return 0
    fi

    local swap_needed=$((RECOMMENDED_TOTAL - total_memory))

    print_color "$YELLOW" "📝 Plan: Add ${swap_needed}GB temporary swap"
    echo "   This will give you ${RECOMMENDED_TOTAL}GB total memory"
    echo ""

    # Check disk space
    local available_disk=$(df -BG / | tail -1 | awk '{print int($4)}')
    if [ $available_disk -lt $((swap_needed + 10)) ]; then
        print_color "$RED" "❌ Insufficient disk space!"
        echo "   Need: ${swap_needed}GB for swap + 10GB buffer"
        echo "   Available: ${available_disk}GB"
        exit 1
    fi

    # User confirmation
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled by user"
        exit 0
    fi

    # Create swap file
    echo ""
    print_color "$YELLOW" "⏳ Creating ${swap_needed}GB swap file (this may take a minute)..."

    # Use dd for better progress indication
    dd if=/dev/zero of="$SWAP_FILE" bs=1G count=$swap_needed status=progress

    # Set permissions
    chmod 600 "$SWAP_FILE"

    # Make swap
    print_color "$YELLOW" "⏳ Formatting as swap..."
    mkswap "$SWAP_FILE"

    # Enable swap
    print_color "$YELLOW" "⏳ Activating swap..."
    swapon "$SWAP_FILE"

    print_color "$GREEN" "✅ Temporary swap created and activated!"

    # Show final status
    show_status

    echo ""
    print_color "$GREEN" "🎉 Success! Your system is ready for Qwen3-80B IPEX optimization"
    print_color "$YELLOW" "⚠️  Remember to remove this swap after model setup with:"
    echo "   $0 remove"
}

# Function to remove temporary swap
remove_swap() {
    echo ""
    print_color "$BLUE" "🗑️  Removing Temporary Swap"
    echo "============================"

    if [ ! -f "$SWAP_FILE" ]; then
        print_color "$RED" "❌ No temporary swap file found at $SWAP_FILE"
        echo "   Either it was already removed or created elsewhere."
        return 0
    fi

    # Check if swap is active
    if swapon --show | grep -q "$SWAP_FILE"; then
        print_color "$YELLOW" "⏳ Deactivating swap..."
        swapoff "$SWAP_FILE"

        if [ $? -eq 0 ]; then
            print_color "$GREEN" "✅ Swap deactivated"
        else
            print_color "$RED" "❌ Failed to deactivate swap"
            echo "   There may be processes using the swap."
            echo "   Try closing some applications and run again."
            exit 1
        fi
    else
        echo "   Swap file exists but is not active"
    fi

    # Remove swap file
    print_color "$YELLOW" "⏳ Removing swap file..."
    rm -f "$SWAP_FILE"

    if [ $? -eq 0 ]; then
        print_color "$GREEN" "✅ Swap file removed successfully"
    else
        print_color "$RED" "❌ Failed to remove swap file"
        exit 1
    fi

    # Show final status
    show_status

    print_color "$GREEN" "✅ Cleanup complete! Temporary swap has been removed."
}

# Function to show help
show_help() {
    echo "Temporary Swap Manager for Qwen3-80B"
    echo "====================================="
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status  - Show current memory and swap status (default)"
    echo "  create  - Create temporary swap if needed"
    echo "  remove  - Remove temporary swap file"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Show status"
    echo "  $0 create            # Create swap if needed"
    echo "  $0 remove            # Remove temporary swap"
    echo ""
    echo "This script helps manage temporary swap space for"
    echo "Qwen3-80B IPEX optimization which needs ~160-200GB RAM."
}

# Main script logic
case "${1:-status}" in
    status)
        show_status
        ;;
    create|add)
        check_root
        create_swap
        ;;
    remove|delete|rm)
        check_root
        remove_swap
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        print_color "$RED" "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
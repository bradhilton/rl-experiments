#!/bin/bash

# fix_numa_topology.sh
# 
# This script identifies all PCI devices in the system with their NUMA node value
# set to -1 (unassigned) and assigns them to NUMA node 1.
#
# Usage: sudo ./fix_numa_topology.sh
#
# Options:
#   --dry-run    Only show devices that would be modified without making changes
#   --help       Display this help message

show_help() {
    echo "Usage: sudo $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dry-run    Only show devices that would be modified without making changes"
    echo "  --help       Display this help message"
    echo
    exit 0
}

# Check if script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (with sudo)" >&2
    echo "Try: sudo $0" >&2
    exit 1
fi

# Process arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --help)
            show_help
            ;;
        --dry-run)
            DRY_RUN=true
            echo "Running in dry-run mode. No changes will be made."
            ;;
        *)
            echo "Unknown argument: $arg"
            show_help
            ;;
    esac
done

# Find devices with numa_node set to -1
echo "Scanning for devices with unassigned NUMA nodes (value: -1)..."
devices=$(find /sys/devices/ -name numa_node -exec printf "%s\t" {} \; -exec cat {} \; | 
          column --table --table-columns device,numa_node | 
          grep -- '-1' | 
          cut -d " " -f 1)

# Count the number of devices found
device_count=$(echo "$devices" | wc -l)
echo "Found $device_count devices with unassigned NUMA nodes."

if [ -z "$devices" ]; then
    echo "No devices found that need topology fixing."
    exit 0
fi

# Show all devices that would be modified
echo -e "\nDevices to be assigned to NUMA node 1:"
echo "$devices"

# Exit if in dry-run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "\nDry run complete. No changes made."
    exit 0
fi

# Confirm before proceeding
echo -e "\nAbout to assign $device_count devices to NUMA node 1."
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Apply changes
echo "Applying NUMA node changes..."
success_count=0
for device in $devices; do
    if echo 1 > "$device" 2>/dev/null; then
        echo "✓ Set $device to NUMA node 1"
        ((success_count++))
    else
        echo "✗ Failed to set $device" >&2
    fi
done

# Summary
echo -e "\nSummary:"
echo "$success_count of $device_count devices successfully updated."

if [ "$success_count" -eq "$device_count" ]; then
    echo "NUMA topology fix completed successfully."
    exit 0
else
    echo "Some devices could not be updated. You may need to check permissions."
    exit 1
fi 
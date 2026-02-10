#!/bin/bash
# Patch script to make log paths configurable at build time
# This script patches vendor code temporarily during build (not committed to git)

set -e

VENDOR_DIR="${1:-vendor/simai/astra-sim-alibabacloud}"
LOG_PATH="${2:-./results/}"
RESULT_PATH="${3:-./results/ncclFlowModel_}"

echo "=== Patching hardcoded paths in $VENDOR_DIR ==="
echo "LOG_PATH: $LOG_PATH"
echo "RESULT_PATH: $RESULT_PATH"

# Patch MockNcclLog.h - replace hardcoded LOG_PATH with conditional define
echo "Patching MockNcclLog.h..."
sed -i.bak 's|#define LOG_PATH  "/etc/astra-sim/"|#ifndef LOG_PATH\n#define LOG_PATH "./results/"\n#endif|g' \
    "$VENDOR_DIR/astra-sim/system/MockNcclLog.h"

# Patch SimAiMain.cc - replace hardcoded RESULT_PATH with conditional define
echo "Patching SimAiMain.cc..."
sed -i.bak 's|#define RESULT_PATH "/etc/astra-sim/results/ncclFlowModel_"|#ifndef RESULT_PATH\n#define RESULT_PATH "./results/ncclFlowModel_"\n#endif|g' \
    "$VENDOR_DIR/astra-sim/network_frontend/phynet/SimAiMain.cc"

# Patch build scripts - replace hardcoded /etc/astra-sim paths
echo "Patching build.sh..."
sed -i.bak 's|SIM_LOG_DIR=/etc/astra-sim|SIM_LOG_DIR=./results|g' \
    "$VENDOR_DIR/build.sh"

echo "Patching build/astra_ns3/build.sh..."
sed -i.bak 's|SIM_LOG_DIR=/etc/astra-sim|SIM_LOG_DIR=./results|g' \
    "$VENDOR_DIR/build/astra_ns3/build.sh"

# Patch SimAI.conf paths to use relative paths
echo "Patching SimAI.conf..."
sed -i.bak 's|/etc/astra-sim/|./|g' \
    "$VENDOR_DIR/inputs/config/SimAI.conf"

echo "=== Patching complete ==="
echo "Note: .bak files created for reference, vendor changes are temporary"

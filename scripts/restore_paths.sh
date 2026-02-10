#!/bin/bash
# Restore original vendor files from backups

set -e

VENDOR_DIR="${1:-vendor/simai/astra-sim-alibabacloud}"

echo "=== Restoring original files in $VENDOR_DIR ==="

# Restore from .bak files
for bakfile in $(find "$VENDOR_DIR" -name "*.bak" 2>/dev/null); do
    original="${bakfile%.bak}"
    echo "Restoring $original"
    mv "$bakfile" "$original"
done

echo "=== Restoration complete ==="

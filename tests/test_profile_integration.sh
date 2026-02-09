#!/bin/bash
set -e  # Exit on error

echo "=== SimAI Profile Integration Tests ==="
echo ""

# Configuration
TEST_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Test 1: Mode 1 - Constant compute times (default)
echo "[Test 1] Mode 1: Constant compute times (default)"
simai generate workload \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    -o test_constant.txt
echo "✓ Generated workload with constant compute times"
echo ""

# Test 2: Mode 2 - Pre-recorded profile (new functionality)
echo "[Test 2] Mode 2: Pre-recorded profile"
echo "  Step 1: Profile GPU kernels..."
simai profile gpu \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --gpu-type H100 \
    -o test_profile.txt
echo "✓ GPU profiling completed"

echo "  Step 2: Generate workload using profile..."
simai generate workload \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --compute-profile test_profile.txt \
    -o test_profiled.txt
echo "✓ Generated workload from profile"

echo "  Step 3: Reuse profile with different config..."
simai generate workload \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --sequence-length 4096 \
    --compute-profile test_profile.txt \
    -o test_profiled_4k.txt
echo "✓ Reused profile for different sequence length"
echo ""

# Test 3: Mode 3 - Live profiling (existing functionality)
echo "[Test 3] Mode 3: Live profiling"
simai generate workload \
    --profile-compute \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    -o test_live_profile.txt
echo "✓ Generated workload with live profiling"
echo ""

# Test 4: Verify outputs exist and are non-empty
echo "[Test 4] Verify output files"
for file in test_constant.txt test_profile.txt test_profiled.txt test_profiled_4k.txt test_live_profile.txt; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing: $file"
        exit 1
    fi
    if [ ! -s "$file" ]; then
        echo "✗ Empty: $file"
        exit 1
    fi
    echo "✓ $file exists and is non-empty"
done
echo ""

# Test 5: Verify profiled workload has reasonable size
echo "[Test 5] Verify workload structures"
constant_lines=$(wc -l < test_constant.txt)
profiled_lines=$(wc -l < test_profiled.txt)
echo "  Constant workload: $constant_lines lines"
echo "  Profiled workload: $profiled_lines lines"
if [ "$profiled_lines" -ge 30 ] && [ "$profiled_lines" -le 100 ]; then
    echo "✓ Profiled workload has reasonable size"
else
    echo "✗ Profiled workload size unexpected: $profiled_lines lines"
    exit 1
fi
echo ""

# Test 6: Verify profiled times are actual values (not -1 placeholders)
echo "[Test 6] Verify profiled times are real (not constant -1)"
# Count lines where the 3rd field (after 2 tabs) is a number > 100000 (real profiled times in nanoseconds)
actual_times=$(awk -F'\t' '$3 > 100000 && $3 != -1' test_profiled.txt | wc -l)
if [ "$actual_times" -gt 5 ]; then
    echo "✓ Found $actual_times operations with real profiled times"
else
    echo "✗ Profiling may not be working (only $actual_times profiled operations found)"
    exit 1
fi
echo ""

# Test 7: Test with MoE model
echo "[Test 7] Test MoE model profiling"
simai profile gpu \
    --framework Megatron \
    --num-gpus 8 \
    --tensor-parallel 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --moe \
    --num-experts 8 \
    --top-k 2 \
    --sequence-parallel \
    --gpu-type H100 \
    -o test_moe_profile.txt
echo "✓ MoE model profiling completed"
echo ""

# Summary
echo "==================================="
echo "All tests passed!"
echo "Test results saved in: $TEST_DIR"
echo "==================================="

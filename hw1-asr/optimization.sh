#!/bin/bash
# apply_optimisations.sh
# Run from: ~/edin-mls-26-spring/hw1-asr/
# Usage: bash apply_optimisations.sh

set -e
TEMPLATE="glm_asr_triton_template"

echo "=== Applying optimisations to $TEMPLATE ==="

# ---- 1. layers.py ----
echo "[1/3] Patching layers.py..."
# Hopper-safe tile sizes (BLOCK_N <= BLOCK_M, avoids Ampere MMA assert)
sed -i 's/TILE_M = [0-9]*/TILE_M = 128/g' $TEMPLATE/layers.py
sed -i 's/TILE_N = [0-9]*/TILE_N = 64/g'  $TEMPLATE/layers.py
sed -i 's/TILE_K = [0-9]*/TILE_K = 32/g'  $TEMPLATE/layers.py
# Auto backend — uses Triton for large M, torch for small
sed -i 's/BACKEND = "torch"/BACKEND = "auto"/' $TEMPLATE/layers.py
sed -i 's/BACKEND = "triton"/BACKEND = "auto"/' $TEMPLATE/layers.py

echo "    layers.py tile sizes: TILE_M=128, TILE_N=64, TILE_K=32 (Hopper-safe)"
grep -n "TILE_M\|TILE_N\|TILE_K\|BACKEND" $TEMPLATE/layers.py | head -10

# ---- 2. attention.py ----
echo "[2/3] Patching attention.py..."
# Raise sequence length cap so FA kernel fires for long KV sequences
sed -i 's/MAX_ATTENTION_DIM = [0-9]*/MAX_ATTENTION_DIM = 4096/' $TEMPLATE/attention.py
# Safe block sizes for Hopper FA kernel: block_m >= block_n
sed -i 's/block_m = [0-9]*/block_m = 64/'  $TEMPLATE/attention.py
sed -i 's/block_n = [0-9]*/block_n = 32/'  $TEMPLATE/attention.py
# Safe warp count for Hopper
sed -i 's/num_warps=[0-9]*/num_warps=4/g'  $TEMPLATE/attention.py
sed -i 's/num_stages=[0-9]*/num_stages=2/g' $TEMPLATE/attention.py

echo "    attention.py: MAX_ATTENTION_DIM=4096, block_m=64, block_n=32"
grep -n "MAX_ATTENTION_DIM\|block_m\|block_n\|num_warps\|num_stages" $TEMPLATE/attention.py | head -10

# ---- 3. rope.py ----
echo "[3/3] Checking rope.py..."
grep -n "use_triton\|_update_cache\|to(device)" $TEMPLATE/rope.py | head -5

echo ""
echo "=== Done! Running benchmark ==="
python benchmark_student.py $TEMPLATE
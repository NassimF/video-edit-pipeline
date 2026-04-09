#!/bin/bash
# GPU and environment sanity check
# Usage: bash scripts/00_check_gpu.sh

BASE_DIR="/workspace/storage_nassim/Video-restyle"
SAM2_CKPT="/workspace/storage_nassim/sam2/checkpoints/sam2.1_hiera_large.pt"
OMNIMATTE_MODEL="/workspace/storage_nassim/gen-omnimatte-public/checkpoints/Wan2.1-Fun-1.3B-InP"
OMNIMATTE_TRANSFORMER="/workspace/storage_nassim/gen-omnimatte-public/checkpoints/wan2.1-v1.0-1.3b-transformer.safetensors"
VACE_WEIGHTS="$BASE_DIR/models/wan_vace/weights/1.3B"

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" = "ok" ]; then
        echo "  [OK]   $label"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL+1))
    fi
}

echo "============================================"
echo " Layer-Aware Video Editing Pipeline — Check"
echo "============================================"

echo ""
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    && echo "" || echo "  [FAIL] nvidia-smi not found"

echo ""
echo "--- CUDA / nvcc ---"
nvcc_ver=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
if [ -n "$nvcc_ver" ]; then
    check "nvcc $nvcc_ver" "ok"
else
    echo "  [WARN] nvcc not in PATH (non-critical — inference only needs CUDA runtime)"
fi

echo ""
echo "--- Model files ---"
[ -f "$SAM2_CKPT" ]                && check "SAM2 checkpoint" "ok"           || check "SAM2 checkpoint ($SAM2_CKPT)" "fail"
[ -d "$OMNIMATTE_MODEL" ]          && check "Gen-omnimatte base model" "ok"   || check "Gen-omnimatte base model ($OMNIMATTE_MODEL)" "fail"
[ -f "$OMNIMATTE_TRANSFORMER" ]    && check "Casper transformer weights" "ok" || check "Casper transformer ($OMNIMATTE_TRANSFORMER)" "fail"
[ -d "$VACE_WEIGHTS" ]             && check "WAN VACE 1.3B weights" "ok"      || check "WAN VACE 1.3B weights ($VACE_WEIGHTS) — run download script" "fail"

echo ""
echo "--- conda environments ---"
conda run -n omnimatte python -c "import torch, sam2, cv2, imageio, omegaconf, mediapy, absl, ml_collections; print('ok')" 2>/dev/null \
    | grep -q "ok" && check "omnimatte env (torch, sam2, mediapy, absl)" "ok" || check "omnimatte env — missing packages" "fail"

conda run -n wan21 python -c "
import sys; sys.path.insert(0,'/workspace/storage_nassim/Wan2.1')
import torch, cv2, imageio, ftfy, decord, easydict
from wan import WanVace
print('ok')
" 2>/dev/null | grep -q "ok" && check "wan21 env (torch, wan, decord, ftfy)" "ok" || check "wan21 env — missing packages" "fail"

echo ""
echo "--- PyTorch GPU access ---"
conda run -n omnimatte python -c "import torch; assert torch.cuda.is_available(); print(f'omnimatte: {torch.cuda.device_count()} GPU(s), torch {torch.__version__}')" 2>/dev/null \
    || echo "  [FAIL] omnimatte env: no CUDA"
conda run -n wan21 python -c "import sys; sys.path.insert(0,'/workspace/storage_nassim/Wan2.1'); import torch; assert torch.cuda.is_available(); print(f'wan21: {torch.cuda.device_count()} GPU(s), torch {torch.__version__}')" 2>/dev/null \
    || echo "  [FAIL] wan21 env: no CUDA"

echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1

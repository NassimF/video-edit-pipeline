#!/bin/bash
# ============================================================
# Layer-Aware Video Editing Pipeline — End-to-End Runner
# ============================================================
# Usage:
#   bash scripts/run_pipeline.sh \
#       --video data/input/videos/my_video.mp4 \
#       --prompt "a person wearing a red shirt" \
#       [--object 0] \
#       [--mask_type alpha] \
#       [--bg_prompt "a sunny beach"] \
#       [--points "427,240"] \
#       [--start_stage 1] \
#       [--end_stage 5]
#
# Stage numbers:
#   1 = SAM2 segmentation
#   2 = Omnimatte Stage 1 (Casper)
#   3 = Omnimatte Stage 2 (RGBA optimization)
#   4 = WAN VACE editing
#   5 = Comparison video
#
# To resume from a checkpoint (e.g., restart Stage 3 after a crash):
#   bash scripts/run_pipeline.sh --video ... --prompt ... --start_stage 3

set -e

# --- Defaults ---
VIDEO=""
PROMPT=""
OBJECT_IDX=0
MASK_TYPE="alpha"
BG_PROMPT="a natural background scene."
POINTS="427,240"
MODEL_SIZE="1.3B"
NUM_STEPS_CASPER=50
NUM_STEPS_OMNI=6000
NUM_STEPS_VACE=50
START_STAGE=1
END_STAGE=5

# --- Parse args ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video)       VIDEO="$2"; shift;;
        --prompt)      PROMPT="$2"; shift;;
        --object)      OBJECT_IDX="$2"; shift;;
        --mask_type)   MASK_TYPE="$2"; shift;;
        --bg_prompt)   BG_PROMPT="$2"; shift;;
        --points)      POINTS="$2"; shift;;
        --model_size)  MODEL_SIZE="$2"; shift;;
        --start_stage) START_STAGE="$2"; shift;;
        --end_stage)   END_STAGE="$2"; shift;;
    esac
    shift
done

if [[ -z "$VIDEO" || -z "$PROMPT" ]]; then
    echo "Usage: $0 --video <path> --prompt <text> [options]"
    echo ""
    echo "Required:"
    echo "  --video     Path to input video"
    echo "  --prompt    Text editing prompt"
    echo ""
    echo "Optional:"
    echo "  --object    Object index to edit (default: 0)"
    echo "  --mask_type alpha|binary (default: alpha)"
    echo "  --bg_prompt Background description for Casper (default: 'a natural background scene.')"
    echo "  --points    SAM2 point prompt as 'x,y' (default: 427,240)"
    echo "  --model_size 1.3B|14B (default: 1.3B)"
    echo "  --start_stage 1-5 (default: 1, to resume from a checkpoint)"
    echo "  --end_stage   1-5 (default: 5)"
    exit 1
fi

BASE_DIR="/workspace/storage_nassim/Video-restyle"
ALPHA_MASK="$BASE_DIR/data/output/03_alpha_masks/alpha_object_${OBJECT_IDX}.mp4"
SAM2_MASK_DIR="$BASE_DIR/data/output/01_sam2_masks/object_${OBJECT_IDX}"

echo "============================================================"
echo " Layer-Aware Video Editing Pipeline"
echo "============================================================"
echo " Video:       $VIDEO"
echo " Prompt:      $PROMPT"
echo " Object:      $OBJECT_IDX"
echo " Mask type:   $MASK_TYPE"
echo " Model:       WAN VACE $MODEL_SIZE"
echo " Stages:      $START_STAGE → $END_STAGE"
echo "============================================================"

# ---- Stage 1: SAM2 Segmentation ----
if [[ $START_STAGE -le 1 && $END_STAGE -ge 1 ]]; then
    echo ""
    echo "▶ Stage 1/5: SAM2 segmentation (GPU 0)..."
    conda run -n omnimatte \
        python "$BASE_DIR/scripts/01_segment.py" \
        --video "$VIDEO" \
        --points "$POINTS" \
        --output "$BASE_DIR/data/output/01_sam2_masks/" \
        --device cuda:0
    echo "  Stage 1 complete."
fi

# ---- Stage 2: Omnimatte Stage 1 (Casper) ----
if [[ $START_STAGE -le 2 && $END_STAGE -ge 2 ]]; then
    echo ""
    echo "▶ Stage 2/5: Omnimatte Stage 1 — Casper (GPU 0)..."
    conda run -n omnimatte \
        python "$BASE_DIR/scripts/02_omnimatte_stage1.py" \
        --video "$VIDEO" \
        --masks "$BASE_DIR/data/output/01_sam2_masks/" \
        --output_solo "$BASE_DIR/data/output/02_solo_videos/" \
        --output_bg "$BASE_DIR/data/output/02_clean_bg/" \
        --bg_prompt "$BG_PROMPT" \
        --device cuda:0 \
        --num_steps "$NUM_STEPS_CASPER"
    echo "  Stage 2 complete."
fi

# ---- Stage 3: Omnimatte Stage 2 (RGBA optimization) ----
if [[ $START_STAGE -le 3 && $END_STAGE -ge 3 ]]; then
    echo ""
    echo "▶ Stage 3/5: Omnimatte Stage 2 — RGBA optimization (GPU 0)..."
    conda run -n omnimatte \
        python "$BASE_DIR/scripts/03_omnimatte_stage2.py" \
        --device cuda:0 \
        --num_steps "$NUM_STEPS_OMNI"
    echo "  Stage 3 complete."
fi

# ---- Stage 4: WAN VACE Editing ----
if [[ $START_STAGE -le 4 && $END_STAGE -ge 4 ]]; then
    echo ""
    echo "▶ Stage 4/5: WAN VACE editing (GPU 1)..."

    VACE_ARGS=(
        --video "$VIDEO"
        --prompt "$PROMPT"
        --output "$BASE_DIR/data/output/04_edited_video/edited.mp4"
        --model_size "$MODEL_SIZE"
        --device cuda:1
        --num_steps "$NUM_STEPS_VACE"
    )

    if [[ "$MASK_TYPE" == "alpha" ]]; then
        VACE_ARGS+=(--alpha "$ALPHA_MASK" --mask_type alpha)
    else
        VACE_ARGS+=(--sam2_mask "$SAM2_MASK_DIR" --mask_type binary)
    fi

    conda run -n wan21 \
        python "$BASE_DIR/scripts/04_wan_vace_edit.py" \
        "${VACE_ARGS[@]}"
    echo "  Stage 4 complete."
fi

# ---- Stage 5: Comparison video ----
if [[ $START_STAGE -le 5 && $END_STAGE -ge 5 ]]; then
    echo ""
    echo "▶ Stage 5/5: Generating comparison video..."
    conda run -n omnimatte \
        python "$BASE_DIR/scripts/05_visualize_results.py" \
        --original "$VIDEO" \
        --edited "$BASE_DIR/data/output/04_edited_video/edited.mp4" \
        --alpha "$ALPHA_MASK" \
        --output "$BASE_DIR/data/output/comparison.mp4"
    echo "  Stage 5 complete."
fi

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo " Edited video:  $BASE_DIR/data/output/04_edited_video/edited.mp4"
echo " Comparison:    $BASE_DIR/data/output/comparison.mp4"
echo "============================================================"

# Layer-Aware Video Editing Pipeline

A modular, training-free pipeline for physically consistent, object-level video editing. 

The pipeline chains two pretrained models: **Generative Omnimatte** decomposes the video into per-object RGBA layers with soft alpha masks that capture the object and its physical effects (shadows, reflections). Those masks then spatially constrain **WAN VACE**, which performs text-guided masked inpainting on the original full video — editing only the masked region while preserving everything else.

---

## Pipeline Overview

```
Input Video
    │
    ▼
┌──────────────────────┐
│  Stage 1: SAM2       │  Point/box prompt on first frame
│  GPU 0               │  → per-frame binary masks per object
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Stage 2: Casper     │  Generative Omnimatte Stage 1
│  GPU 0  (~20 min)    │  → solo videos + clean background
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Stage 3: Omnimatte  │  RGBA layer optimization
│  GPU 0  (~10 min)    │  → soft alpha mask per object
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Stage 4: WAN VACE   │  Text-guided masked inpainting
│  GPU 1  (~5 min)     │  on the ORIGINAL full video
└──────────────────────┘
    │
    ▼
  Edited Video
```

**Key design choice:** WAN VACE receives the original full video (not isolated layers) with the Omnimatte alpha mask as a spatial constraint. This keeps the full scene context in view — matching VACE's training distribution — so edits are photorealistic and physically grounded.

---

## Hardware

| Component | Spec |
|---|---|
| GPUs | 2× NVIDIA A100-SXM4-80GB (NVLink) |
| VRAM | 160 GB total |
| OS | Ubuntu 22.04.4 LTS |
| CUDA | 12.2 |
| Target resolution | 480p (832×480) |

---

## Models Used

| Model | Role | Weights |
|---|---|---|
| [SAM2.1-Large](https://github.com/facebookresearch/sam2) | Object segmentation | `sam2.1_hiera_large.pt` |
| [Generative Omnimatte](https://gen-omnimatte.github.io) (Wan2.1-based Casper) | Video layer decomposition | `Wan2.1-Fun-1.3B-InP` + Casper transformer |
| [WAN VACE 1.3B](https://github.com/Wan-Video/Wan2.1) | Text-guided masked inpainting | `Wan-AI/Wan2.1-VACE-1.3B` |

---

## Setup

### Prerequisites

Clone the required model repos (expected at these paths on the DGX server):

```bash
# SAM2
git clone https://github.com/facebookresearch/sam2.git /workspace/storage_nassim/sam2
cd /workspace/storage_nassim/sam2
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
conda run -n omnimatte pip install -e /workspace/storage_nassim/sam2

# Generative Omnimatte
git clone https://github.com/gen-omnimatte/gen-omnimatte-public.git /workspace/storage_nassim/gen-omnimatte-public
# Download Casper weights to: gen-omnimatte-public/checkpoints/
# See https://gen-omnimatte.github.io for weight download links

# Wan2.1 (includes VACE)
git clone https://github.com/Wan-Video/Wan2.1.git /workspace/storage_nassim/Wan2.1
```

### Required patch: gen-omnimatte SAM2 path

`gen-omnimatte-public/omnimatte/utils.py` contains a hardcoded path to the original author's server. Patch it before running Stage 3:

```python
# In gen-omnimatte-public/omnimatte/utils.py, line ~344
# Change:
def __init__(self, sam_model='large', SAM_DIR="/nfshomes/yclee/disk/projects/sam2"):
# To:
def __init__(self, sam_model='large', SAM_DIR="/workspace/storage_nassim/sam2"):
```

### Input video resolution

Both WAN VACE and Casper (Omnimatte Stage 1) require **832×480** resolution. Other resolutions (e.g. 640×384) are not supported and will raise errors. Stage 4 auto-resizes the input video if needed, but it is recommended to provide a native 832×480 video to avoid any quality loss from upscaling.

### Conda Environments

Two environments are needed — specs are in `envs/`:

```bash
# Stage 1-3: SAM2 + Omnimatte
conda env create -f envs/omnimatte_env.yaml

# Stage 4: WAN VACE
conda env create -f envs/wan_env.yaml
```

### WAN VACE Weights

```bash
conda run -n wan21 python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-VACE-1.3B',
                  local_dir='models/wan_vace/weights/1.3B')
"
```

### Verify Setup

```bash
bash scripts/00_check_gpu.sh
```

Expected: 6/6 checks pass — 2 A100s, all model files found, both conda envs working.

---

## Usage

### Full pipeline (end-to-end)

```bash
bash scripts/run_pipeline.sh \
    --video data/input/videos/my_video.mp4 \
    --prompt "a person wearing a red shirt" \
    --boxes "345,140,425,330"   # xmin,ymin,xmax,ymax around the object in frame 1
```

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Input video path |
| `--prompt` | *(required)* | Text editing prompt |
| `--boxes` | — | SAM2 box prompt (xmin,ymin,xmax,ymax); preferred over `--points` for full-body coverage |
| `--points` | `427,240` | SAM2 point prompt (x,y on object, frame 1); less reliable than boxes |
| `--object` | `0` | Object index to edit |
| `--mask_type` | `alpha` | `alpha` (Omnimatte soft) or `binary` (SAM2) |
| `--bg_prompt` | `a natural background scene.` | Background description for Casper |
| `--model_size` | `1.3B` | `1.3B` (~10 GB VRAM) or `14B` (~70 GB VRAM) |
| `--start_stage` | `1` | Resume from this stage number after a crash |

### Stage-by-stage (recommended for first run)

```bash
# Stage 1 — Segment the object (~1 min)
# Box prompts (--boxes xmin,ymin,xmax,ymax) are more reliable than point prompts
# for full-body coverage. Inspect the output mask video and re-run with adjusted
# coordinates if needed.
conda run -n omnimatte python scripts/01_segment.py \
    --video data/input/videos/my_video.mp4 \
    --boxes "345,140,425,330"
# Inspect: data/output/01_sam2_masks/object_0_mask.mp4
# Adjust --boxes if needed, then re-run.

# Stage 2 — Casper: solo videos + clean BG (~20 min)
conda run -n omnimatte python scripts/02_omnimatte_stage1.py \
    --video data/input/videos/my_video.mp4 \
    --bg_prompt "a sunny outdoor scene"
# Inspect: data/output/02_solo_videos/  and  data/output/02_clean_bg/

# Stage 3 — RGBA optimization (~10 min)
# Requires the SAM2 path patch in gen-omnimatte-public/omnimatte/utils.py (see Setup)
conda run -n omnimatte python scripts/03_omnimatte_stage2.py
# Inspect: data/output/03_alpha_masks/alpha_object_0.mp4

# Stage 4 — WAN VACE edit (~5 min)
# Input video is auto-resized to 832x480 if needed (VACE hard requirement)
conda run -n wan21 python scripts/04_wan_vace_edit.py \
    --video data/input/videos/my_video.mp4 \
    --alpha data/output/03_alpha_masks/alpha_object_0.mp4 \
    --prompt "a person wearing a red shirt"

# Stage 5 — Comparison video
conda run -n omnimatte python scripts/05_visualize_results.py \
    --original data/input/videos/my_video.mp4 \
    --edited data/output/04_edited_video/edited.mp4 \
    --alpha data/output/03_alpha_masks/alpha_object_0.mp4
```

### Object replacement (binary mask)

For replacing objects entirely (e.g. bike → car), skip Omnimatte and use the SAM2 binary mask:

```bash
bash scripts/run_pipeline.sh \
    --video data/input/videos/my_video.mp4 \
    --prompt "a red car driving down the road" \
    --mask_type binary \
    --start_stage 4
```

---

## Output Structure

```
data/output/
├── 01_sam2_masks/
│   ├── frames/                    # Extracted JPEG frames (kept for Stage 2)
│   ├── object_0/                  # Per-frame binary PNG masks
│   └── object_0_mask.mp4          # Visual mask video for inspection
├── 02_solo_videos/
│   └── solo_object_0.mp4          # Object 0 isolated (others inpainted)
├── 02_clean_bg/
│   └── clean_bg.mp4               # All objects removed
├── casper_workspace/              # gen-omnimatte sequence dir (internal)
├── casper_outputs/                # Raw Casper output MP4s (internal)
├── 03_rgba_layers/pipeline_seq/
│   ├── fg00_alpha.mp4             # Soft alpha mask (key output)
│   ├── fg00_rgba_checker.mp4
│   └── fg00_visualization.mp4
├── 03_alpha_masks/
│   └── alpha_object_0.mp4         # Alpha mask → input to Stage 4
├── 04_edited_video/
│   └── edited.mp4                 # Final edited video
└── comparison.mp4                 # Side-by-side: original | mask | edited
```

---

## GPU Memory & Runtime

| Stage | Model | GPU | VRAM | Time (85 frames, 480p, 2 objects) |
|---|---|---|---|---|
| SAM2 segmentation | SAM2.1-Large | GPU 0 | ~4 GB | ~1–2 min |
| Omnimatte Stage 1 (Casper) | Wan2.1-Fun-1.3B | GPU 0 | ~25 GB | ~20–25 min |
| Omnimatte Stage 2 (RGBA) | Optimization | GPU 0 | ~8 GB | ~25–30 min |
| WAN VACE 1.3B | WAN VACE 1.3B | GPU 1 | ~10 GB | ~8–10 min |

**Total (1.3B mode, 2 objects): ~55–70 minutes per video**

---

## Project Structure

```
video-edit-pipeline/
├── configs/pipeline_config.yaml   # All tunable parameters + model paths
├── scripts/
│   ├── 00_check_gpu.sh            # Environment sanity check
│   ├── 01_segment.py              # SAM2 segmentation
│   ├── 02_omnimatte_stage1.py     # Casper: solo videos + clean BG
│   ├── 03_omnimatte_stage2.py     # RGBA layer optimization
│   ├── 04_wan_vace_edit.py        # WAN VACE text-guided editing
│   ├── 05_visualize_results.py    # Side-by-side comparison
│   └── run_pipeline.sh            # End-to-end runner
├── envs/
│   ├── omnimatte_env.yaml         # Conda env for Stages 1-3
│   └── wan_env.yaml               # Conda env for Stage 4
├── models/wan_vace/weights/       # WAN VACE weights (gitignored)
├── data/input/videos/             # Place input videos here (gitignored)
├── data/output/                   # All stage outputs (gitignored)
└── PLAN.md                        # Detailed implementation plan
```

---

## Roadmap

- [x] Phase 1: Environment setup, model downloads, all pipeline scripts
- [ ] Phase 2: End-to-end test on real video, parameter tuning
- [ ] Phase 3: Multi-object editing
- [ ] Phase 4: Gradio web UI
- [ ] Phase 5: SAM2-only binary mask path (faster, no Omnimatte)

---

## References

- [Generative Omnimatte](https://gen-omnimatte.github.io) — CVPR 2025
- [WAN VACE](https://github.com/Wan-Video/Wan2.1) — ICCV 2025
- [SAM2](https://github.com/facebookresearch/sam2) — Meta AI
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Alibaba

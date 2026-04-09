# Layer-Aware Video Editing Pipeline — Implementation Plan

## Hardware & Environment

- **Server:** DGX A100
- **OS:** Ubuntu 22.04.4 LTS (Jammy)
- **GPUs:** 2× NVIDIA A100-SXM4-80GB (NVLink interconnect)
- **Total VRAM:** 160 GB
- **Driver:** 535.230.02
- **CUDA:** 12.2
- **Target resolution:** 480p (854×480) for initial testing; scalable to 720p

---

## Pipeline Overview

```
Input Video + Object Masks
         │
         ▼
┌─────────────────────┐
│  Stage 1: SAM2      │  → Binary object masks (per object, per frame)
│  GPU 0              │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: Omnimatte │  → Solo videos (Casper) + Clean background
│  Stage 1 — GPU 0    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3: Omnimatte │  → RGBA layers per object (RGB + soft alpha mask)
│  Stage 2 — GPU 0    │
└─────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Stage 4: WAN VACE MV2V Inpainting       │
│  Input: original video + alpha mask      │
│         + text prompt                    │
│  GPU 1 (1.3B) or GPU 0+1 (14B)          │
│  Output: edited video                    │
└──────────────────────────────────────────┘
```

**Key design decision:** WAN VACE receives the original full video (not isolated layers)
with the Omnimatte alpha mask as a spatial constraint. This ensures:
- WAN sees natural, complete scene context (matches its training distribution)
- Edits are physically constrained to the masked region
- Shadow/reflection regions are included in the alpha mask and update correctly

---

## Project Directory Structure

```
~/video-edit-pipeline/
├── PLAN.md                        # This file
├── configs/
│   └── pipeline_config.yaml       # All tunable parameters
├── scripts/
│   ├── 01_segment.py              # SAM2: generate per-frame object masks
│   ├── 02_omnimatte_stage1.py     # Casper: solo videos + clean BG
│   ├── 03_omnimatte_stage2.py     # RGBA layer optimization
│   ├── 04_wan_vace_edit.py        # WAN VACE: text-guided masked editing
│   ├── 05_visualize_results.py    # Side-by-side comparison video
│   └── run_pipeline.sh            # End-to-end runner
├── models/
│   ├── sam2/                      # SAM2 weights
│   ├── omnimatte/                 # Generative Omnimatte (Casper) weights
│   ├── depthcrafter/              # DepthCrafter weights (for layer ordering)
│   └── wan_vace/                  # WAN VACE — code repos + weights
│       ├── VACE/                  # ali-vilab/VACE code repo (cloned)
│       ├── Wan2.1/                # Wan-Video/Wan2.1 code repo (cloned)
│       └── weights/               # Model weights from HuggingFace
│           ├── 1.3B/              # WAN VACE 1.3B weights
│           └── 14B/               # WAN VACE 14B weights
├── data/
│   ├── input/
│   │   ├── videos/                # Raw input .mp4 files
│   │   └── reference_masks/       # Optional: hand-drawn first-frame masks
│   └── output/
│       ├── 01_sam2_masks/         # Per-frame binary masks from SAM2
│       ├── 02_solo_videos/        # Solo videos per object from Casper
│       ├── 02_clean_bg/           # Clean background video from Casper
│       ├── 03_rgba_layers/        # RGBA omnimatte layers per object
│       ├── 03_alpha_masks/        # Extracted soft alpha masks
│       └── 04_edited_video/       # Final WAN VACE output
├── envs/
│   ├── omnimatte_env.yaml         # Conda env spec for Omnimatte
│   └── wan_env.yaml               # Conda env spec for WAN VACE
└── logs/
    └── pipeline.log               # Runtime logs
```

---

## Phase 1: Environment Setup

### Step 1.1 — Check system dependencies

```bash
# Confirmed: Ubuntu 22.04.4 LTS, CUDA 12.2, Driver 535.230.02
# Verify nvcc and conda are accessible
nvcc --version
conda --version

# Install system deps if missing
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0
git lfs install
```

### Step 1.2 — Install Miniconda (if not already installed)

```bash
# Check first — DGX servers often have conda pre-installed
conda --version

# If not found, install Miniconda for Ubuntu 22.04
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash && source ~/.bashrc
```

### Step 1.3 — Create Omnimatte environment

```bash
conda create -n omnimatte python=3.10 -y
conda activate omnimatte

# PyTorch with CUDA 12.2
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install diffusers==0.27.2 transformers==4.40.0 accelerate==0.30.0
pip install opencv-python pillow imageio imageio-ffmpeg
pip install scipy einops tqdm omegaconf
pip install segment-anything-2
```

### Step 1.4 — Create WAN VACE environment

```bash
conda create -n wan_vace python=3.10 -y
conda activate wan_vace

# PyTorch with CUDA 12.2
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# WAN VACE dependencies
pip install diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0
pip install huggingface_hub opencv-python pillow imageio imageio-ffmpeg
pip install easydict omegaconf tqdm

# For multi-GPU inference with WAN 14B (optional but recommended)
pip install "xfuser>=0.4.1"
```

---

## Phase 2: Model Downloads

### Step 2.1 — Download SAM2

```bash
conda activate omnimatte
cd ~/video-edit-pipeline/models

git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .

# Download SAM2 weights (use sam2.1_hiera_large for best quality)
mkdir -p ../sam2/checkpoints
wget -P ../sam2/checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Step 2.2 — Download Generative Omnimatte

```bash
cd ~/video-edit-pipeline/models

# Clone the Generative Omnimatte repository
git clone https://github.com/gen-omnimatte/gen-omnimatte.git omnimatte
cd omnimatte && pip install -e .

# Download model weights from the project page
# NOTE: Check https://gen-omnimatte.github.io for the latest weight download link.
# If weights require request access, download manually and place in:
# ~/video-edit-pipeline/models/omnimatte/checkpoints/casper_cogvideox.safetensors
```

> **Note on Omnimatte availability:** The Generative Omnimatte paper (CVPR 2025)
> uses Lumiere internally (Google-internal model), but the public release uses a
> CogVideoX-based version of Casper. If the official release is not yet fully
> public, use **OmnimatteZero** as a fallback:
> `git clone https://github.com/OmnimatteZero/omnimatte-zero.git`
> OmnimatteZero uses WAN2.1 directly for training-free omnimatte decomposition
> and is fully compatible with this pipeline.

### Step 2.3 — Download DepthCrafter (for layer ordering)

```bash
cd ~/video-edit-pipeline/models

git clone https://github.com/DepthCrafter/DepthCrafter.git depthcrafter
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/DepthCrafter',
                  local_dir='depthcrafter/weights')
"
```

### Step 2.4 — Clone and install WAN VACE

```bash
conda activate wan_vace
cd ~/video-edit-pipeline

# Clone the VACE code repo (inference scripts, pipeline, preprocessors)
git clone https://github.com/ali-vilab/VACE.git models/wan_vace/VACE
cd models/wan_vace/VACE
pip install -e .
cd ~/video-edit-pipeline

# Also clone the main WAN2.1 repo (required by VACE as a dependency)
git clone https://github.com/Wan-Video/Wan2.1.git models/wan_vace/Wan2.1
cd models/wan_vace/Wan2.1
pip install -e .
cd ~/video-edit-pipeline

# Download WAN VACE 1.3B model weights (recommended for initial testing — ~8GB VRAM)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-VACE-1.3B',
                  local_dir='models/wan_vace/weights/1.3B')
"

# Download WAN VACE 14B model weights (for production quality — needs ~70GB VRAM)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-VACE-14B',
                  local_dir='models/wan_vace/weights/14B')
"
```

---

## Phase 3: Configuration

Create `configs/pipeline_config.yaml`:

```yaml
# Pipeline Configuration
# Tuned for 2× A100-SXM4-80GB, CUDA 12.2, 480p initial testing

input:
  video_path: "data/input/videos/my_video.mp4"
  fps: 24                        # Target FPS (Omnimatte works best at 24)
  max_frames: 80                 # Omnimatte processes up to 80 frames (Casper limit)
  resolution: "480p"             # 854×480 for initial testing

gpu:
  omnimatte_device: "cuda:0"     # GPU 0 for all Omnimatte stages
  wan_device: "cuda:1"           # GPU 1 for WAN VACE 1.3B
  wan_multi_gpu: false           # Set true to use GPU 0+1 for WAN 14B
  wan_model_size: "1.3B"         # "1.3B" or "14B"

sam2:
  model_path: "models/sam2/checkpoints/sam2.1_hiera_large.pt"
  model_cfg: "sam2.1_hiera_l.yaml"
  # Prompting mode: "point" (click on object) or "mask" (draw rough mask)
  prompt_mode: "point"

omnimatte:
  model_path: "models/omnimatte"
  num_ddpm_steps: 256            # Casper inference steps (default from paper)
  batch_size: 4                  # Process 4 layers in parallel (bg + 3 objects max)
  output_resolution: [480, 854]  # [H, W] — 480p

wan_vace:
  code_path: "models/wan_vace/VACE"
  model_path_1b: "models/wan_vace/weights/1.3B"
  model_path_14b: "models/wan_vace/weights/14B"
  num_inference_steps: 50        # Denoising steps (50 is good balance)
  guidance_scale: 6.0
  mask_dilation_px: 5            # Slightly dilate alpha mask before passing to VACE
                                 # Helps catch edge pixels missed by Omnimatte

editing:
  # Mask type selection
  # "alpha"  — soft alpha from Omnimatte Stage 2 (use for appearance edits: recoloring, retexturing)
  # "binary" — SAM2 binary mask (use for object replacement: bike → car)
  mask_type: "alpha"

  # Text prompt for the edit
  edit_prompt: "a person wearing a red shirt"

  # Which object index to edit (0-indexed, matches SAM2 mask order)
  target_object_index: 0

output:
  dir: "data/output"
  save_intermediate: true        # Save output of each stage (useful for debugging)
  video_format: "mp4"
  comparison_video: true         # Generate side-by-side original vs edited
```

---

## Phase 4: Script Implementation

### Script 01 — SAM2 segmentation

`scripts/01_segment.py`:

```python
"""
Stage 1: Generate per-frame object masks using SAM2.
Usage:
    python scripts/01_segment.py \
        --config configs/pipeline_config.yaml \
        --video data/input/videos/my_video.mp4 \
        --output data/output/01_sam2_masks/
        
Prompting:
    - For point prompts: click on the object in the first frame
    - For mask prompts: draw a rough mask on the first frame

Outputs:
    - data/output/01_sam2_masks/object_0/  (binary mask PNGs per frame)
    - data/output/01_sam2_masks/object_1/  (if multiple objects)
    - data/output/01_sam2_masks/object_0_mask.mp4  (mask video for inspection)
"""

import os, sys, cv2, torch, argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor

def main(args):
    cfg = OmegaConf.load(args.config)
    device = cfg.gpu.omnimatte_device  # SAM2 runs on GPU 0

    # Build SAM2 video predictor
    predictor = build_sam2_video_predictor(
        cfg.sam2.model_cfg,
        cfg.sam2.model_path,
        device=device
    )

    # Extract frames from video
    video_dir = Path(args.output) / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"ffmpeg -i {args.video} -q:v 2 -vf scale=854:480 {video_dir}/%05d.jpg -y")

    frame_paths = sorted(video_dir.glob("*.jpg"))
    print(f"Extracted {len(frame_paths)} frames")

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video_dir))

        # Add object prompts on first frame
        # For point prompt: provide (x, y) coordinates on the object
        # Edit these coordinates to match your video
        prompts = [
            {"type": "point", "points": [[427, 240]], "labels": [1]},  # object 0
            # Add more objects here if needed:
            # {"type": "point", "points": [[200, 300]], "labels": [1]},  # object 1
        ]

        for obj_id, prompt in enumerate(prompts):
            if prompt["type"] == "point":
                _, _, _ = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array(prompt["points"]),
                    labels=np.array(prompt["labels"]),
                )

        # Propagate masks across all frames
        output_dir = Path(args.output)
        for obj_id in range(len(prompts)):
            (output_dir / f"object_{obj_id}").mkdir(parents=True, exist_ok=True)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(object_ids, masks):
                mask_np = (mask[0].cpu().numpy() > 0).astype(np.uint8) * 255
                out_path = output_dir / f"object_{obj_id}" / f"{frame_idx:05d}.png"
                cv2.imwrite(str(out_path), mask_np)

    print(f"✅ SAM2 masks saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="data/output/01_sam2_masks")
    main(parser.parse_args())
```

### Script 02 — Omnimatte Stage 1 (Casper)

`scripts/02_omnimatte_stage1.py`:

```python
"""
Stage 2: Run Generative Omnimatte Stage 1 (Casper object-effect removal).
Produces solo videos (one per object, with others removed + background inpainted)
and a clean background video.

Usage:
    python scripts/02_omnimatte_stage1.py \
        --config configs/pipeline_config.yaml \
        --video data/input/videos/my_video.mp4 \
        --masks data/output/01_sam2_masks/ \
        --output data/output/02_solo_videos/

Outputs:
    - data/output/02_solo_videos/solo_object_0.mp4   (Boy 1 only, Boy 2 inpainted)
    - data/output/02_solo_videos/solo_object_1.mp4   (Boy 2 only, Boy 1 inpainted)
    - data/output/02_clean_bg/clean_bg.mp4           (all objects removed)

Memory note:
    Casper (CogVideoX-based) uses ~25-35 GB VRAM on GPU 0.
    It processes up to 80 frames at a time.
    For longer videos, temporal multidiffusion is applied automatically.
"""

import os, torch, argparse
from pathlib import Path
from omegaconf import OmegaConf

def main(args):
    cfg = OmegaConf.load(args.config)
    device = cfg.gpu.omnimatte_device  # GPU 0

    # Import Omnimatte after setting device
    # Adjust import path based on actual Omnimatte repo structure
    sys.path.insert(0, str(Path(cfg.omnimatte.model_path)))
    from omnimatte.pipeline import OmnimattePipeline

    pipeline = OmnimattePipeline.from_pretrained(
        cfg.omnimatte.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    mask_dir = Path(args.masks)
    output_dir = Path(args.output)
    bg_dir = Path("data/output/02_clean_bg")
    output_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # Discover all object mask directories
    object_dirs = sorted(mask_dir.glob("object_*"))
    n_objects = len(object_dirs)
    print(f"Found {n_objects} objects to process")

    # Run Casper for each solo video and clean background
    # Trimask convention: 1=preserve, 0=remove, 0.5=uncertain background
    for target_obj_idx in range(n_objects):
        print(f"Processing solo video for object {target_obj_idx}...")
        result = pipeline.run_stage1(
            video_path=args.video,
            mask_dirs=[str(d) for d in object_dirs],
            target_object=target_obj_idx,
            num_steps=cfg.omnimatte.num_ddpm_steps,
            resolution=tuple(cfg.omnimatte.output_resolution),
        )
        out_path = output_dir / f"solo_object_{target_obj_idx}.mp4"
        result.save(str(out_path))
        print(f"  Saved: {out_path}")

    # Clean background (remove all objects)
    print("Processing clean background...")
    bg_result = pipeline.run_background(
        video_path=args.video,
        mask_dirs=[str(d) for d in object_dirs],
        num_steps=cfg.omnimatte.num_ddpm_steps,
        resolution=tuple(cfg.omnimatte.output_resolution),
    )
    bg_result.save(str(bg_dir / "clean_bg.mp4"))
    print(f"  Saved: {bg_dir / 'clean_bg.mp4'}")
    print("✅ Omnimatte Stage 1 complete.")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", default="data/output/01_sam2_masks")
    parser.add_argument("--output", default="data/output/02_solo_videos")
    main(parser.parse_args())
```

### Script 03 — Omnimatte Stage 2 (RGBA optimization)

`scripts/03_omnimatte_stage2.py`:

```python
"""
Stage 3: Run Generative Omnimatte Stage 2 (RGBA layer optimization).
Takes (solo video, clean BG) pairs and produces RGBA omnimatte layers
with soft alpha masks that capture object + associated effects.

Usage:
    python scripts/03_omnimatte_stage2.py \
        --config configs/pipeline_config.yaml \
        --solo_dir data/output/02_solo_videos/ \
        --bg data/output/02_clean_bg/clean_bg.mp4 \
        --output data/output/03_rgba_layers/

Outputs:
    - data/output/03_rgba_layers/rgba_object_0.npz  (RGB + alpha arrays)
    - data/output/03_alpha_masks/alpha_object_0.mp4 (visualized alpha for inspection)

The alpha mask is the key output used by WAN VACE in the next stage.
It is a soft (float) mask, not binary — values range from 0.0 to 1.0.
"""

import sys, torch, argparse, numpy as np
from pathlib import Path
from omegaconf import OmegaConf

def main(args):
    cfg = OmegaConf.load(args.config)
    device = cfg.gpu.omnimatte_device  # GPU 0

    sys.path.insert(0, str(Path(cfg.omnimatte.model_path)))
    from omnimatte.optimization import OmnimatteOptimizer

    solo_dir = Path(args.solo_dir)
    output_dir = Path(args.output)
    alpha_dir = Path("data/output/03_alpha_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    solo_videos = sorted(solo_dir.glob("solo_object_*.mp4"))

    for solo_path in solo_videos:
        obj_idx = int(solo_path.stem.split("_")[-1])
        print(f"Optimizing RGBA layer for object {obj_idx}...")

        optimizer = OmnimatteOptimizer(
            solo_video=str(solo_path),
            bg_video=args.bg,
            device=device,
            num_iterations=20000,       # From paper default
            resolution=tuple(cfg.omnimatte.output_resolution),
        )
        rgba_layer = optimizer.run()

        # Save RGBA data
        out_path = output_dir / f"rgba_object_{obj_idx}.npz"
        np.savez(str(out_path),
                 rgb=rgba_layer["rgb"],
                 alpha=rgba_layer["alpha"])

        # Save alpha as video for visual inspection
        alpha_video_path = alpha_dir / f"alpha_object_{obj_idx}.mp4"
        optimizer.save_alpha_video(rgba_layer["alpha"], str(alpha_video_path))

        print(f"  RGBA saved: {out_path}")
        print(f"  Alpha video: {alpha_video_path}")

    print("✅ Omnimatte Stage 2 complete.")
    print(f"Alpha masks ready at: data/output/03_alpha_masks/")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--solo_dir", default="data/output/02_solo_videos")
    parser.add_argument("--bg", default="data/output/02_clean_bg/clean_bg.mp4")
    parser.add_argument("--output", default="data/output/03_rgba_layers")
    main(parser.parse_args())
```

### Script 04 — WAN VACE editing

`scripts/04_wan_vace_edit.py`:

```python
"""
Stage 4: Run WAN VACE MV2V (Masked Video-to-Video) editing.
Takes the original full video, the alpha mask from Omnimatte Stage 2,
and a text prompt. Edits only within the masked region.

Usage (recoloring — alpha mask):
    python scripts/04_wan_vace_edit.py \
        --config configs/pipeline_config.yaml \
        --video data/input/videos/my_video.mp4 \
        --rgba_layer data/output/03_rgba_layers/rgba_object_0.npz \
        --prompt "a person wearing a red shirt" \
        --output data/output/04_edited_video/edited.mp4

Usage (object replacement — binary mask):
    python scripts/04_wan_vace_edit.py \
        --config configs/pipeline_config.yaml \
        --video data/input/videos/my_video.mp4 \
        --sam2_mask data/output/01_sam2_masks/object_0/ \
        --mask_type binary \
        --prompt "a car driving down the road" \
        --output data/output/04_edited_video/edited.mp4

GPU memory:
    - WAN VACE 1.3B: ~8 GB → GPU 1 alone (CUDA_VISIBLE_DEVICES=1)
    - WAN VACE 14B: ~70 GB → both GPUs via xfuser tensor parallelism

VACE mask convention: 1=edit this region, 0=preserve this region
(opposite of some inpainting conventions — be careful)
"""

import sys, os, torch, argparse, numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf

def load_alpha_as_vace_mask(rgba_npz_path, dilation_px=5):
    """Convert Omnimatte soft alpha to VACE binary mask with slight dilation."""
    data = np.load(rgba_npz_path)
    alpha = data["alpha"]  # shape: (T, H, W), float32, range [0,1]

    # Binarize: pixels with alpha > 0.1 are part of the object+effects
    binary = (alpha > 0.1).astype(np.uint8)

    # Dilate slightly to catch edge pixels
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        binary = np.stack([cv2.dilate(f, kernel) for f in binary])

    return binary  # (T, H, W), uint8, 0 or 1

def load_sam2_mask(mask_dir):
    """Load SAM2 binary masks for object replacement."""
    mask_paths = sorted(Path(mask_dir).glob("*.png"))
    masks = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        masks.append((m > 127).astype(np.uint8))
    return np.stack(masks)  # (T, H, W)

def main(args):
    cfg = OmegaConf.load(args.config)

    # Determine mask type and GPU assignment
    mask_type = args.mask_type or cfg.editing.mask_type
    model_size = cfg.wan_vace.wan_model_size
    use_multi_gpu = cfg.gpu.wan_multi_gpu

    if use_multi_gpu:
        # Both GPUs for 14B model via xfuser
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        device = "cuda"
    else:
        # Single GPU 1 for 1.3B model
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = "cuda:0"  # Remapped to GPU 1 via CUDA_VISIBLE_DEVICES

    # Load VACE mask
    if mask_type == "alpha" and args.rgba_layer:
        print("Using soft alpha mask from Omnimatte Stage 2...")
        vace_mask = load_alpha_as_vace_mask(
            args.rgba_layer,
            dilation_px=cfg.editing.mask_dilation_px
        )
    elif mask_type == "binary" and args.sam2_mask:
        print("Using binary SAM2 mask for object replacement...")
        vace_mask = load_sam2_mask(args.sam2_mask)
    else:
        raise ValueError("Provide either --rgba_layer (alpha) or --sam2_mask (binary)")

    # Load VACE model
    model_path = (cfg.wan_vace.model_path_14b
                  if model_size == "14B"
                  else cfg.wan_vace.model_path_1b)

    print(f"Loading WAN VACE {model_size} from {model_path}...")

    # Import VACE pipeline
    # Adjust import based on actual VACE repo structure
    sys.path.insert(0, "models/wan_vace")
    from vace.vace_pipeline import VACEPipeline

    pipeline = VACEPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    if use_multi_gpu:
        from xfuser import xFuserArgs
        pipeline = pipeline.to_distributed()
    else:
        pipeline = pipeline.to(device)

    # Run VACE MV2V inpainting
    print(f"Running WAN VACE edit...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Mask type: {mask_type}")
    print(f"  Model: WAN VACE {model_size}")

    result = pipeline(
        prompt=args.prompt,
        src_video=args.video,
        src_mask=vace_mask,        # (T, H, W) binary mask: 1=edit, 0=preserve
        num_inference_steps=cfg.wan_vace.num_inference_steps,
        guidance_scale=cfg.wan_vace.guidance_scale,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"✅ Edited video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--video", required=True, help="Original input video")
    parser.add_argument("--rgba_layer", help="Path to .npz from Omnimatte Stage 2")
    parser.add_argument("--sam2_mask", help="Path to SAM2 mask dir (for object replacement)")
    parser.add_argument("--mask_type", choices=["alpha", "binary"],
                        help="Override config mask_type")
    parser.add_argument("--prompt", required=True, help="Text editing prompt")
    parser.add_argument("--output", default="data/output/04_edited_video/edited.mp4")
    main(parser.parse_args())
```

### Script 05 — Visualization

`scripts/05_visualize_results.py`:

```python
"""
Stage 5: Generate side-by-side comparison video (original vs edited).
Also generates a debug video showing the alpha mask overlay.

Usage:
    python scripts/05_visualize_results.py \
        --original data/input/videos/my_video.mp4 \
        --edited data/output/04_edited_video/edited.mp4 \
        --alpha data/output/03_alpha_masks/alpha_object_0.mp4 \
        --output data/output/comparison.mp4
"""

import cv2, numpy as np, argparse
from pathlib import Path

def make_comparison(original_path, edited_path, alpha_path, output_path):
    cap_orig = cv2.VideoCapture(original_path)
    cap_edit = cv2.VideoCapture(edited_path)
    cap_alpha = cv2.VideoCapture(alpha_path) if alpha_path else None

    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Side-by-side: [original | alpha overlay | edited]
    panels = 3 if cap_alpha else 2
    out_w = w * panels
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, h))

    label_font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret1, f_orig = cap_orig.read()
        ret2, f_edit = cap_edit.read()
        if not ret1 or not ret2:
            break

        cv2.putText(f_orig, "ORIGINAL", (10, 30), label_font, 1, (255,255,255), 2)
        cv2.putText(f_edit, "EDITED", (10, 30), label_font, 1, (255,255,255), 2)

        if cap_alpha:
            _, f_alpha = cap_alpha.read()
            f_alpha_colored = cv2.applyColorMap(f_alpha, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(f_orig, 0.6, f_alpha_colored, 0.4, 0)
            cv2.putText(overlay, "ALPHA MASK", (10, 30), label_font, 1, (255,255,255), 2)
            frame = np.hstack([f_orig, overlay, f_edit])
        else:
            frame = np.hstack([f_orig, f_edit])

        writer.write(frame)

    writer.release()
    print(f"✅ Comparison video saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--output", default="data/output/comparison.mp4")
    args = parser.parse_args()
    make_comparison(args.original, args.edited, args.alpha, args.output)
```

### End-to-end runner

`scripts/run_pipeline.sh`:

```bash
#!/bin/bash
# End-to-end pipeline runner
# Usage: bash scripts/run_pipeline.sh --video my_video.mp4 --prompt "a person wearing a red shirt"

set -e  # Exit on any error

VIDEO=""
PROMPT=""
OBJECT_IDX=0
MASK_TYPE="alpha"
CONFIG="configs/pipeline_config.yaml"

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) VIDEO="$2"; shift;;
        --prompt) PROMPT="$2"; shift;;
        --object) OBJECT_IDX="$2"; shift;;
        --mask_type) MASK_TYPE="$2"; shift;;
        --config) CONFIG="$2"; shift;;
    esac
    shift
done

if [[ -z "$VIDEO" || -z "$PROMPT" ]]; then
    echo "Usage: $0 --video <path> --prompt <text> [--object 0] [--mask_type alpha|binary]"
    exit 1
fi

echo "============================================"
echo " Layer-Aware Video Editing Pipeline"
echo "============================================"
echo " Video:   $VIDEO"
echo " Prompt:  $PROMPT"
echo " Object:  $OBJECT_IDX"
echo " Mask:    $MASK_TYPE"
echo "============================================"

# Step 1: SAM2 segmentation (GPU 0)
echo ""
echo "▶ Step 1/4: SAM2 segmentation..."
conda run -n omnimatte python scripts/01_segment.py \
    --config "$CONFIG" \
    --video "$VIDEO" \
    --output data/output/01_sam2_masks/

# Step 2: Omnimatte Stage 1 (GPU 0)
echo ""
echo "▶ Step 2/4: Omnimatte Stage 1 (Casper)..."
conda run -n omnimatte python scripts/02_omnimatte_stage1.py \
    --config "$CONFIG" \
    --video "$VIDEO" \
    --masks data/output/01_sam2_masks/ \
    --output data/output/02_solo_videos/

# Step 3: Omnimatte Stage 2 (GPU 0)
echo ""
echo "▶ Step 3/4: Omnimatte Stage 2 (RGBA optimization)..."
conda run -n omnimatte python scripts/03_omnimatte_stage2.py \
    --config "$CONFIG" \
    --solo_dir data/output/02_solo_videos/ \
    --bg data/output/02_clean_bg/clean_bg.mp4 \
    --output data/output/03_rgba_layers/

# Step 4: WAN VACE editing (GPU 1)
echo ""
echo "▶ Step 4/4: WAN VACE editing..."
RGBA_LAYER="data/output/03_rgba_layers/rgba_object_${OBJECT_IDX}.npz"
SAM2_MASK="data/output/01_sam2_masks/object_${OBJECT_IDX}/"

conda run -n wan_vace python scripts/04_wan_vace_edit.py \
    --config "$CONFIG" \
    --video "$VIDEO" \
    --rgba_layer "$RGBA_LAYER" \
    --sam2_mask "$SAM2_MASK" \
    --mask_type "$MASK_TYPE" \
    --prompt "$PROMPT" \
    --output data/output/04_edited_video/edited.mp4

# Step 5: Generate comparison video
echo ""
echo "▶ Generating comparison video..."
conda run -n omnimatte python scripts/05_visualize_results.py \
    --original "$VIDEO" \
    --edited data/output/04_edited_video/edited.mp4 \
    --alpha "data/output/03_alpha_masks/alpha_object_${OBJECT_IDX}.mp4" \
    --output data/output/comparison.mp4

echo ""
echo "============================================"
echo "✅ Pipeline complete!"
echo " Edited video: data/output/04_edited_video/edited.mp4"
echo " Comparison:   data/output/comparison.mp4"
echo "============================================"
```

---

## Phase 5: GPU Memory & Runtime Estimates

| Stage | Model | GPU | VRAM Est. | Runtime (80 frames, 480p) |
|---|---|---|---|---|
| SAM2 segmentation | SAM2.1-Large | GPU 0 | ~4 GB | ~1 min |
| Omnimatte Stage 1 | CogVideoX-Casper | GPU 0 | ~28 GB | ~15–25 min |
| Omnimatte Stage 2 | Optimization | GPU 0 | ~8 GB | ~8–12 min |
| WAN VACE 1.3B | WAN VACE 1.3B | GPU 1 | ~8 GB | ~3–5 min |
| WAN VACE 14B | WAN VACE 14B | GPU 0+1 | ~70 GB | ~12–20 min |

**Total estimated runtime per video (1.3B mode):** ~30–45 minutes
**Total estimated runtime per video (14B mode):** ~55–80 minutes

Stages are run **sequentially** to avoid OOM errors. GPU 0 handles all
Omnimatte work; GPU 1 handles WAN VACE. The A100 NVLink interconnect
is used when both GPUs are engaged for WAN 14B multi-GPU inference.

---

## Phase 6: Testing Protocol

### Test 1 — Verify each stage independently

```bash
# Test SAM2 only (quick, <2 min)
bash scripts/00_check_gpu.sh
conda activate omnimatte
python scripts/01_segment.py --video data/input/videos/test_clip.mp4

# Inspect masks before proceeding
vlc data/output/01_sam2_masks/object_0_mask.mp4
```

```bash
# Test Omnimatte Stage 1 only (~20 min)
conda activate omnimatte
python scripts/02_omnimatte_stage1.py \
    --video data/input/videos/test_clip.mp4 \
    --masks data/output/01_sam2_masks/

# Inspect solo video quality before proceeding
vlc data/output/02_solo_videos/solo_object_0.mp4
```

```bash
# Test WAN VACE edit only (requires pre-computed alpha mask)
conda activate wan_vace
python scripts/04_wan_vace_edit.py \
    --video data/input/videos/test_clip.mp4 \
    --rgba_layer data/output/03_rgba_layers/rgba_object_0.npz \
    --prompt "a person wearing a red shirt"
```

### Test 2 — Full pipeline end-to-end

```bash
# Simple recoloring test
bash scripts/run_pipeline.sh \
    --video data/input/videos/test_clip.mp4 \
    --prompt "a person wearing a red shirt" \
    --object 0 \
    --mask_type alpha
```

### Test 3 — Object replacement (binary mask)

```bash
bash scripts/run_pipeline.sh \
    --video data/input/videos/bike_video.mp4 \
    --prompt "a red car driving down the road" \
    --object 0 \
    --mask_type binary
```

---

## Phase 7: Gradio Web UI (Future Phase)

Once the CLI pipeline is working, add a web UI by creating:

`app.py` — Gradio interface that:
- Accepts video upload
- Shows first frame for point-based SAM2 prompting (click on object)
- Accepts text prompt for the edit
- Runs the pipeline as a background job
- Displays progress updates per stage
- Shows side-by-side comparison when done

Launch with:
```bash
conda activate wan_vace
python app.py --server_port 7860 --share
```

---

## Known Issues & Mitigations

**Issue 1 — Omnimatte weights not yet public**
The CogVideoX-based Casper may require manual request to the authors.
*Fallback:* Use OmnimatteZero (training-free, uses WAN directly) to get
solo videos and alpha masks. The rest of the pipeline is unchanged.

**Issue 2 — WAN VACE mask convention**
VACE uses 1=edit, 0=preserve. Omnimatte alpha is continuous [0,1].
Script 04 handles binarization with a 0.1 threshold and dilation.
If edits bleed outside the object, lower the threshold to 0.2.
If edits miss edge pixels, increase `mask_dilation_px` in config.

**Issue 3 — Videos longer than 80 frames**
Omnimatte's Casper processes 80 frames at a time. For longer videos,
temporal multidiffusion is applied automatically by the pipeline.
WAN VACE 1.3B also has an 81-frame limit. Use sliding window approach
or trim test videos to under 80 frames initially.

**Issue 4 — SAM2 mask drift on fast-moving objects**
If SAM2 loses track mid-video, use the `--reference_mask` option
to provide additional keyframe prompts at frames where drift occurs.

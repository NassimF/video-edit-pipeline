"""
Stage 1: Generate per-frame object masks using SAM2.

Extracts frames from the input video, runs SAM2 video predictor with point/box
prompts on the first frame, propagates masks across all frames, and saves
per-frame binary PNGs plus a visual mask video.

Usage (point prompt):
    conda run -n omnimatte python scripts/01_segment.py \\
        --video data/input/videos/my_video.mp4 \\
        --points "427,240" \\
        --output data/output/01_sam2_masks/

Usage (multiple objects):
    conda run -n omnimatte python scripts/01_segment.py \\
        --video data/input/videos/my_video.mp4 \\
        --points "427,240" "150,300" \\
        --output data/output/01_sam2_masks/

Usage (box prompt — xmin,ymin,xmax,ymax):
    conda run -n omnimatte python scripts/01_segment.py \\
        --video data/input/videos/my_video.mp4 \\
        --boxes "100,50,300,400" \\
        --output data/output/01_sam2_masks/

Outputs:
    <output>/object_0/00000.png  ... per-frame binary masks (0 or 255)
    <output>/object_0_mask.mp4   ... visual mask video for inspection
    <output>/frames/             ... extracted JPEG frames (kept for Stage 2)
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path

# SAM2 was installed from /root/sam2 into the omnimatte env
SAM2_REPO = "/workspace/storage_nassim/sam2"
sys.path.insert(0, SAM2_REPO)

import torch
from sam2.build_sam import build_sam2_video_predictor


def extract_frames(video_path: str, out_dir: Path, resolution: str = "832x480") -> int:
    """Extract frames as JPEGs. Returns frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = resolution.split("x")
    cmd = (
        f'ffmpeg -i "{video_path}" -q:v 2 '
        f'-vf "scale={w}:{h}" '
        f'"{out_dir}/%05d.jpg" -y -loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}")
    frames = sorted(out_dir.glob("*.jpg"))
    print(f"Extracted {len(frames)} frames → {out_dir}")
    return len(frames)


def save_mask_video(mask_dir: Path, out_path: Path, fps: float = 16.0) -> None:
    """Save binary mask PNGs as a coloured video for visual inspection."""
    frames = sorted(mask_dir.glob("*.png"))
    if not frames:
        return
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
        isColor=True,
    )
    for f in frames:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        # Green channel for masks — easier to see on dark backgrounds
        colour = np.zeros((h, w, 3), dtype=np.uint8)
        colour[:, :, 1] = gray
        writer.write(colour)
    writer.release()


def main(args):
    device = args.device

    # ---- Build SAM2 predictor ----
    # SAM2 uses Hydra config system: config_file is a path relative to the
    # package's internal configs/ directory, NOT an absolute filesystem path.
    cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "/workspace/storage_nassim/sam2/checkpoints/sam2.1_hiera_large.pt"
    print(f"Loading SAM2 from {checkpoint}")
    predictor = build_sam2_video_predictor(cfg_path, checkpoint, device=device)

    # ---- Extract frames ----
    output_dir = Path(args.output)
    frame_dir = output_dir / "frames"
    n_frames = extract_frames(args.video, frame_dir, resolution=args.resolution)

    # ---- Parse prompts ----
    point_prompts = []
    for pt in (args.points or []):
        x, y = pt.split(",")
        point_prompts.append({"x": int(x), "y": int(y)})

    box_prompts = []
    for bx in (args.boxes or []):
        vals = [int(v) for v in bx.split(",")]
        box_prompts.append(vals)  # [xmin, ymin, xmax, ymax]

    if not point_prompts and not box_prompts:
        print("WARNING: No prompts provided. Using default centre-frame point.")
        # Read first frame to get dimensions
        sample = cv2.imread(str(sorted(frame_dir.glob("*.jpg"))[0]))
        h, w = sample.shape[:2]
        point_prompts = [{"x": w // 2, "y": h // 2}]

    n_objects = max(len(point_prompts), len(box_prompts), 1)

    # ---- Run SAM2 propagation ----
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(state)

        for obj_id, prompt in enumerate(point_prompts):
            pts = np.array([[prompt["x"], prompt["y"]]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=pts,
                labels=labels,
            )

        for obj_id, box in enumerate(box_prompts):
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32),
            )

        # Create output dirs for each object
        for obj_id in range(n_objects):
            (output_dir / f"object_{obj_id}").mkdir(parents=True, exist_ok=True)

        print("Propagating masks across all frames...")
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(object_ids, masks):
                mask_np = (mask[0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                out_path = output_dir / f"object_{obj_id}" / f"{frame_idx:05d}.png"
                cv2.imwrite(str(out_path), mask_np)

    # ---- Save visual mask videos ----
    for obj_id in range(n_objects):
        mask_dir = output_dir / f"object_{obj_id}"
        video_path = output_dir / f"object_{obj_id}_mask.mp4"
        save_mask_video(mask_dir, video_path)
        print(f"Mask video saved: {video_path}")

    print(f"\nSAM2 segmentation complete.")
    print(f"  Masks: {output_dir}/object_*/")
    print(f"  Frames: {frame_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 video segmentation")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--points", nargs="+", default=None,
        help="Point prompts as 'x,y' (one per object)"
    )
    parser.add_argument(
        "--boxes", nargs="+", default=None,
        help="Box prompts as 'xmin,ymin,xmax,ymax' (one per object)"
    )
    parser.add_argument(
        "--output", default="data/output/01_sam2_masks",
        help="Output directory"
    )
    parser.add_argument(
        "--resolution", default="832x480",
        help="WxH to resize frames (default: 832x480 for Wan2.1)"
    )
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())

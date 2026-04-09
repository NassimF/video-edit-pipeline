"""
Stage 2: Run Generative Omnimatte Stage 1 (Casper) via gen-omnimatte-public.

Converts our SAM2 PNG masks + input video into the gen-omnimatte sequence format,
then calls inference/wan2.1_fun/predict_v2v.py in "solo" mode.

In solo mode with N objects, Casper runs N+1 passes:
    fg_id=-1  → clean background (all objects inpainted away)
    fg_id=0   → solo video with object 0 (object 1,2,... removed)
    fg_id=1   → solo video with object 1 (object 0,2,... removed)
    etc.

These outputs are later consumed by Stage 2 (reconstruct_omnimatte.py).
For a single object, Casper runs only fg_id=-1 (inpainted background);
the original video is used as the solo video in that case.

Usage:
    conda run -n omnimatte python scripts/02_omnimatte_stage1.py \\
        --video data/input/videos/my_video.mp4 \\
        --masks data/output/01_sam2_masks/ \\
        --bg_prompt "a sunny beach with sand and ocean"

Outputs (persistent across stages):
    data/output/casper_workspace/pipeline_seq/   ← sequence dir (kept for Stage 3)
    data/output/casper_outputs/<seq>-fg=*.mp4    ← Casper outputs (kept for Stage 3)

Convenience copies for inspection:
    data/output/02_solo_videos/solo_object_0.mp4
    data/output/02_clean_bg/clean_bg.mp4
"""

import os
import sys
import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path

OMNIMATTE_REPO = "/workspace/storage_nassim/gen-omnimatte-public"
OMNIMATTE_MODEL = "/workspace/storage_nassim/gen-omnimatte-public/checkpoints/Wan2.1-Fun-1.3B-InP"
OMNIMATTE_TRANSFORMER = (
    "/workspace/storage_nassim/gen-omnimatte-public/checkpoints/"
    "wan2.1-v1.0-1.3b-transformer.safetensors"
)

BASE_DIR = "/workspace/storage_nassim/Video-restyle"
SEQ_NAME = "pipeline_seq"


def png_dir_to_mask_video(png_dir: Path, out_path: Path, fps: float = 16.0) -> None:
    """Convert per-frame binary PNG masks to a grayscale MP4 for gen-omnimatte."""
    frames = sorted(png_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG masks found in {png_dir}")
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
        isColor=False,
    )
    for f in frames:
        writer.write(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
    writer.release()


def setup_sequence_dir(
    seq_dir: Path,
    video_path: str,
    mask_dirs: list,
    bg_prompt: str,
    fps: float = 16.0,
) -> None:
    """
    Build gen-omnimatte sequence directory:
        seq_dir/input_video.mp4
        seq_dir/mask_00.mp4, mask_01.mp4, ...
        seq_dir/prompt.json  →  {"bg": "<bg_prompt>"}
    """
    seq_dir.mkdir(parents=True, exist_ok=True)

    dst_video = seq_dir / "input_video.mp4"
    if not dst_video.exists():
        shutil.copy2(video_path, dst_video)
        print(f"  Copied input video → {dst_video.name}")

    for idx, mask_dir in enumerate(mask_dirs):
        out_mask = seq_dir / f"mask_{idx:02d}.mp4"
        if not out_mask.exists():
            print(f"  Converting object {idx} masks → {out_mask.name}")
            png_dir_to_mask_video(Path(mask_dir), out_mask, fps=fps)

    prompt_path = seq_dir / "prompt.json"
    with open(prompt_path, "w") as f:
        json.dump({"bg": bg_prompt}, f, indent=2)
    print(f"  Wrote prompt.json: bg='{bg_prompt}'")


def run_casper(
    data_rootdir: Path,
    casper_outputs: Path,
    device_id: str = "0",
    gpu_memory_mode: str = "model_cpu_offload",
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> None:
    """Call gen-omnimatte predict_v2v.py via subprocess in solo mode."""
    casper_outputs.mkdir(parents=True, exist_ok=True)

    script = os.path.join(OMNIMATTE_REPO, "inference", "wan2.1_fun", "predict_v2v.py")
    cmd = [
        sys.executable, script,
        f"--config.data.data_rootdir={data_rootdir}",
        f"--config.experiment.run_seqs={SEQ_NAME}",
        f"--config.experiment.save_path={casper_outputs}",
        "--config.experiment.matting_mode=solo",
        f"--config.video_model.model_name={OMNIMATTE_MODEL}",
        f"--config.video_model.transformer_path={OMNIMATTE_TRANSFORMER}",
        f"--config.video_model.config_path={OMNIMATTE_REPO}/config/wan2.1/wan_civitai.yaml",
        f"--config.system.gpu_memory_mode={gpu_memory_mode}",
        f"--config.video_model.num_inference_steps={num_inference_steps}",
        f"--config.video_model.guidance_scale={guidance_scale}",
        "--config.experiment.skip_if_exists=False",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_id

    print(f"\nRunning Casper (solo mode)...")
    print(f"  Save path: {casper_outputs}")
    result = subprocess.run(cmd, cwd=OMNIMATTE_REPO, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Casper failed with return code {result.returncode}")


def copy_outputs_for_inspection(
    casper_outputs: Path,
    solo_dir: Path,
    bg_dir: Path,
) -> None:
    """Copy Casper outputs to named directories for easy inspection."""
    solo_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    all_mp4s = sorted(casper_outputs.glob(f"{SEQ_NAME}-fg=*.mp4"))
    # Exclude _tuple.mp4 (side-by-side debug renders)
    all_mp4s = [f for f in all_mp4s if "_tuple" not in f.name]

    for src in all_mp4s:
        name = src.name
        # fg=-1 is the clean background
        if "fg=-1" in name:
            dst = bg_dir / "clean_bg.mp4"
            shutil.copy2(src, dst)
            print(f"  Clean BG → {dst}")
        else:
            # Extract object index from filename: pipeline_seq-fg=00-0001.mp4
            try:
                fg_str = name.split("fg=")[1].split("-")[0]
                obj_idx = int(fg_str)
                dst = solo_dir / f"solo_object_{obj_idx}.mp4"
                shutil.copy2(src, dst)
                print(f"  Solo object {obj_idx} → {dst}")
            except (IndexError, ValueError):
                print(f"  Skipped (could not parse fg index): {name}")


def main(args):
    mask_base = Path(args.masks)
    object_dirs = sorted(mask_base.glob("object_*"))
    # Only keep dirs that contain PNGs (exclude files like mask videos)
    object_dirs = [d for d in object_dirs if d.is_dir() and list(d.glob("*.png"))]

    if not object_dirs:
        raise FileNotFoundError(
            f"No object_*/PNG dirs found in {args.masks}. Run 01_segment.py first."
        )

    print(f"Found {len(object_dirs)} object(s): {[d.name for d in object_dirs]}")

    workspace = Path(BASE_DIR) / "data" / "output" / "casper_workspace"
    casper_outputs = Path(BASE_DIR) / "data" / "output" / "casper_outputs"
    seq_dir = workspace / SEQ_NAME

    # 1. Set up sequence directory
    print("\n[1/3] Setting up gen-omnimatte sequence directory...")
    setup_sequence_dir(
        seq_dir=seq_dir,
        video_path=args.video,
        mask_dirs=object_dirs,
        bg_prompt=args.bg_prompt,
        fps=args.fps,
    )

    # 2. Run Casper
    print("\n[2/3] Running Casper...")
    run_casper(
        data_rootdir=workspace,
        casper_outputs=casper_outputs,
        device_id=args.device.replace("cuda:", ""),
        gpu_memory_mode=args.gpu_memory_mode,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    # 3. Copy outputs for inspection
    print("\n[3/3] Copying outputs for inspection...")
    copy_outputs_for_inspection(
        casper_outputs=casper_outputs,
        solo_dir=Path(args.output_solo),
        bg_dir=Path(args.output_bg),
    )

    print("\nOmnimatte Stage 1 complete.")
    print(f"  Workspace (needed by Stage 3): {workspace}")
    print(f"  Casper outputs (needed by Stage 3): {casper_outputs}")
    print(f"  Solo videos (inspect): {args.output_solo}")
    print(f"  Clean BG (inspect): {args.output_bg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnimatte Stage 1 (Casper)")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--masks", default="data/output/01_sam2_masks",
        help="Dir with object_0/, object_1/, ... SAM2 PNG mask subdirs"
    )
    parser.add_argument(
        "--output_solo", default="data/output/02_solo_videos",
        help="Dir for solo video copies (inspection only)"
    )
    parser.add_argument(
        "--output_bg", default="data/output/02_clean_bg",
        help="Dir for clean BG video copy (inspection only)"
    )
    parser.add_argument(
        "--bg_prompt", default="a natural background scene.",
        help="Text description of the background scene"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu_memory_mode", default="model_cpu_offload",
        choices=["model_full_load", "model_cpu_offload",
                 "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=16.0)
    main(parser.parse_args())

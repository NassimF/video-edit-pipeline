"""
Stage 4: Run WAN VACE masked video-to-video editing.

Takes the original full video + alpha mask (from Omnimatte Stage 2, or binary
SAM2 mask for object replacement) + a text prompt, and edits only the masked
region while preserving the rest of the scene.

Key design: WAN VACE receives the ORIGINAL full video (not isolated layers)
with the alpha mask as a spatial constraint. This ensures:
  - VACE sees natural, complete scene context (matches training distribution)
  - Edits are confined to the masked region
  - Shadow/reflection regions included in alpha are edited correctly

VACE mask convention: 1 = edit this region, 0 = preserve this region

Usage (appearance edit — Omnimatte alpha mask):
    conda run -n wan21 python scripts/04_wan_vace_edit.py \\
        --video data/input/videos/my_video.mp4 \\
        --alpha data/output/03_alpha_masks/alpha_object_0.mp4 \\
        --prompt "a person wearing a red shirt" \\
        --output data/output/04_edited_video/edited.mp4

Usage (object replacement — SAM2 binary mask):
    conda run -n wan21 python scripts/04_wan_vace_edit.py \\
        --video data/input/videos/my_video.mp4 \\
        --sam2_mask data/output/01_sam2_masks/object_0/ \\
        --mask_type binary \\
        --prompt "a car driving down the road" \\
        --output data/output/04_edited_video/edited.mp4

GPU memory:
    WAN VACE 1.3B: ~10 GB → use GPU 1 (cuda:1) while Stage 1-3 used GPU 0
    WAN VACE 14B:  ~70 GB → needs both GPUs (set CUDA_VISIBLE_DEVICES=0,1)
"""

import os
import sys
import cv2
import argparse
import tempfile
import shutil
import numpy as np
from pathlib import Path

# WAN VACE is in the Wan2.1 repo (no separate install needed)
WAN_REPO = "/workspace/storage_nassim/Wan2.1"
sys.path.insert(0, WAN_REPO)

BASE_DIR = "/workspace/storage_nassim/Video-restyle"


def load_alpha_mask_from_video(
    alpha_video_path: str,
    dilation_px: int = 5,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Load Omnimatte alpha mask video and convert to binary VACE mask.
    Returns (T, H, W) uint8 array with 1=edit, 0=preserve.
    """
    cap = cv2.VideoCapture(alpha_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Alpha video is stored as RGB (all 3 channels identical)
        # Normalise to [0,1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()

    if not frames:
        raise ValueError(f"Could not read frames from alpha video: {alpha_video_path}")

    alpha = np.stack(frames, axis=0)  # (T, H, W) float32 in [0, 1]

    # Binarise: pixels where alpha > threshold are part of the edit region
    binary = (alpha > threshold).astype(np.uint8)

    # Dilate slightly to capture edge pixels that Omnimatte may have missed
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        binary = np.stack([cv2.dilate(f, kernel) for f in binary])

    print(f"Alpha mask loaded: {binary.shape}, "
          f"edit region = {binary.mean()*100:.1f}% of pixels")
    return binary  # (T, H, W), uint8, values 0 or 1


def load_sam2_binary_mask(
    mask_dir: str,
    dilation_px: int = 0,
) -> np.ndarray:
    """Load SAM2 per-frame PNG masks as a binary (T, H, W) array."""
    mask_paths = sorted(Path(mask_dir).glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")

    frames = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frames.append((m > 127).astype(np.uint8))

    binary = np.stack(frames, axis=0)  # (T, H, W)

    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        binary = np.stack([cv2.dilate(f, kernel) for f in binary])

    print(f"SAM2 binary mask loaded: {binary.shape}")
    return binary


def resize_video(src_path: str, out_path: str, width: int, height: int, fps: float = 16.0) -> None:
    """Re-encode video to target resolution using ffmpeg."""
    ret = os.system(
        f'ffmpeg -i "{src_path}" -vf "scale={width}:{height}" '
        f'-r {fps} -c:v libx264 -pix_fmt yuv420p -y "{out_path}" -loglevel error'
    )
    if ret != 0:
        raise RuntimeError(f"ffmpeg resize failed for {src_path}")


def save_mask_as_video(mask: np.ndarray, out_path: str, fps: float = 16.0) -> None:
    """Save (T, H, W) binary mask array as a grayscale MP4 for VACE via ffmpeg."""
    T, H, W = mask.shape
    tmpdir = Path(tempfile.mkdtemp(prefix="mask_frames_"))
    try:
        for t in range(T):
            cv2.imwrite(str(tmpdir / f"{t:05d}.png"), (mask[t] * 255).astype(np.uint8))
        os.system(
            f'ffmpeg -framerate {fps} -i "{tmpdir}/%05d.png" '
            f'-c:v libx264 -pix_fmt yuv420p -y "{out_path}" -loglevel error'
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def align_frame_count(mask: np.ndarray, target_frames: int) -> np.ndarray:
    """Trim or repeat-pad mask to match target frame count (must be 4n+1)."""
    T = mask.shape[0]
    if T >= target_frames:
        return mask[:target_frames]
    # Pad by repeating last frame
    pad = np.stack([mask[-1]] * (target_frames - T), axis=0)
    return np.concatenate([mask, pad], axis=0)


def nearest_valid_frame_count(n: int, max_frames: int = 81) -> int:
    """Return nearest 4n+1 value <= max_frames."""
    n = min(n, max_frames)
    # Round down to 4n+1
    return ((n - 1) // 4) * 4 + 1


def main(args):
    import torch

    # Determine mask type
    mask_type = args.mask_type
    if mask_type == "alpha" and not args.alpha:
        raise ValueError("--alpha path required when --mask_type alpha")
    if mask_type == "binary" and not args.sam2_mask:
        raise ValueError("--sam2_mask path required when --mask_type binary")

    # Load mask
    if mask_type == "alpha":
        mask = load_alpha_mask_from_video(
            args.alpha,
            dilation_px=args.mask_dilation_px,
            threshold=args.alpha_threshold,
        )
    else:
        mask = load_sam2_binary_mask(args.sam2_mask, dilation_px=args.mask_dilation_px)

    # Align frame count to 4n+1 constraint
    target_frames = nearest_valid_frame_count(mask.shape[0], max_frames=args.frame_num)
    mask = align_frame_count(mask, target_frames)
    print(f"Frame count aligned to {target_frames} (4n+1 constraint)")

    # Save mask as temporary video for VACE
    with tempfile.NamedTemporaryFile(suffix="_mask.mp4", delete=False) as tmp:
        tmp_mask_path = tmp.name
    save_mask_as_video(mask, tmp_mask_path, fps=args.fps)

    # Set GPU
    device_id = args.device.replace("cuda:", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = "cuda:0"  # Remapped via CUDA_VISIBLE_DEVICES

    # Load WAN VACE
    from wan import WanVace
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

    model_size = args.model_size
    if model_size == "14B":
        config_key = "vace-14B"
        weights_dir = os.path.join(BASE_DIR, "models/wan_vace/weights/14B")
    else:
        config_key = "vace-1.3B"
        weights_dir = os.path.join(BASE_DIR, "models/wan_vace/weights/1.3B")

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"WAN VACE weights not found: {weights_dir}\n"
            "Run: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('Wan-AI/Wan2.1-VACE-{model_size}', local_dir='{weights_dir}')\""
        )

    print(f"\nLoading WAN VACE {model_size} from {weights_dir}...")
    wan_vace = WanVace(
        config=WAN_CONFIGS[config_key],
        checkpoint_dir=weights_dir,
        device_id=0,  # remapped via CUDA_VISIBLE_DEVICES
        rank=0,
        t5_cpu=args.offload_model,
    )

    # Parse output size  (WAN uses width×height)
    size_str = args.size  # e.g., "832*480"
    if "*" in size_str:
        w_str, h_str = size_str.split("*")
        size = (int(w_str), int(h_str))
    else:
        size = SIZE_CONFIGS[size_str]

    # VACE loads videos at their native resolution — resize input to target size
    # so output matches the alpha mask resolution (832×480)
    W_target, H_target = size
    cap_check = cv2.VideoCapture(args.video)
    vid_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_check.release()

    input_video_path = args.video
    tmp_resized_video = None
    if vid_w != W_target or vid_h != H_target:
        print(f"Resizing input video from {vid_w}x{vid_h} to {W_target}x{H_target}...")
        tmp_resized = tempfile.NamedTemporaryFile(suffix="_resized.mp4", delete=False)
        tmp_resized_video = tmp_resized.name
        tmp_resized.close()
        resize_video(args.video, tmp_resized_video, W_target, H_target, fps=args.fps)
        input_video_path = tmp_resized_video

    # Prepare source (video + mask)
    print(f"Preparing source video + mask...")
    src_videos, src_masks, src_ref_images = wan_vace.prepare_source(
        src_video=[input_video_path],
        src_mask=[tmp_mask_path],
        src_ref_images=[None],
        num_frames=target_frames,
        image_size=size,
        device=torch.device(device),
    )

    # Generate
    print(f"\nRunning WAN VACE generation...")
    print(f"  Prompt:  '{args.prompt}'")
    print(f"  Size:    {size}")
    print(f"  Frames:  {target_frames}")
    print(f"  Steps:   {args.num_steps}")
    print(f"  Guide:   {args.guidance_scale}")

    output_video = wan_vace.generate(
        input_prompt=args.prompt,
        input_frames=src_videos,
        input_masks=src_masks,
        input_ref_images=src_ref_images,
        size=size,
        frame_num=target_frames,
        context_scale=args.context_scale,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.num_steps,
        guide_scale=args.guidance_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=args.offload_model,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # output_video is a tensor (C, T, H, W) in [0,1] or [-1,1]
    # Convert and write with OpenCV
    if isinstance(output_video, torch.Tensor):
        video_np = output_video.cpu().float().numpy()
        if video_np.min() < 0:
            video_np = (video_np + 1.0) / 2.0
        video_np = np.clip(video_np, 0, 1)
        # (C, T, H, W) → (T, H, W, C)
        video_np = video_np.transpose(1, 2, 3, 0)
    else:
        video_np = np.array(output_video)

    T, H, W, C = video_np.shape
    tmpdir = Path(tempfile.mkdtemp(prefix="edited_frames_"))
    try:
        for t in range(T):
            frame_bgr = cv2.cvtColor(
                (video_np[t] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(tmpdir / f"{t:05d}.png"), frame_bgr)
        os.system(
            f'ffmpeg -framerate {args.fps} -i "{tmpdir}/%05d.png" '
            f'-c:v libx264 -pix_fmt yuv420p -y "{output_path}" -loglevel error'
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Cleanup temp files
    os.unlink(tmp_mask_path)
    if tmp_resized_video and os.path.exists(tmp_resized_video):
        os.unlink(tmp_resized_video)

    print(f"\nEdited video saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN VACE masked video editing")
    parser.add_argument("--video", required=True, help="Original input video path")

    # Mask inputs (choose one)
    parser.add_argument(
        "--alpha", default=None,
        help="Omnimatte alpha mask video path (for appearance edits)"
    )
    parser.add_argument(
        "--sam2_mask", default=None,
        help="SAM2 mask dir with per-frame PNGs (for object replacement)"
    )
    parser.add_argument(
        "--mask_type", choices=["alpha", "binary"], default="alpha",
        help="Which mask to use: 'alpha' (Omnimatte) or 'binary' (SAM2)"
    )

    # Edit parameters
    parser.add_argument("--prompt", required=True, help="Text editing prompt")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument(
        "--output", default="data/output/04_edited_video/edited.mp4"
    )

    # Model
    parser.add_argument(
        "--model_size", choices=["1.3B", "14B"], default="1.3B"
    )
    parser.add_argument("--device", default="cuda:1",
                        help="GPU for VACE (use cuda:1 to leave GPU 0 free)")

    # Generation parameters
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--context_scale", type=float, default=1.0)
    parser.add_argument(
        "--sample_solver", choices=["unipc", "dpm++"], default="unipc"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_num", type=int, default=81,
                        help="Max frames to process (must be 4n+1)")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument(
        "--size", default="832*480",
        help="Output size as WxH (e.g., '832*480') or SIZE_CONFIGS key"
    )

    # Mask processing
    parser.add_argument("--mask_dilation_px", type=int, default=5)
    parser.add_argument("--alpha_threshold", type=float, default=0.1)
    parser.add_argument(
        "--offload_model", action="store_true", default=True,
        help="Offload model to CPU between steps (saves VRAM)"
    )

    main(parser.parse_args())

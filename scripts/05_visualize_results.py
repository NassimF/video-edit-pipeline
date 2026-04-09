"""
Stage 5: Generate side-by-side comparison video.

Creates a 3-panel video: [original | alpha mask overlay | edited]
Optionally shows a 2-panel version if no alpha mask is provided.

Usage:
    conda run -n omnimatte python scripts/05_visualize_results.py \\
        --original data/input/videos/my_video.mp4 \\
        --edited data/output/04_edited_video/edited.mp4 \\
        --alpha data/output/03_alpha_masks/alpha_object_0.mp4 \\
        --output data/output/comparison.mp4
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def read_video_frames(path: str) -> tuple:
    """Read all frames from a video. Returns (frames_list, fps, w, h)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps, w, h


def resize_frames(frames: list, target_w: int, target_h: int) -> list:
    """Resize all frames to (target_w, target_h)."""
    return [cv2.resize(f, (target_w, target_h)) for f in frames]


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a white text label with black shadow to the top-left of a frame."""
    f = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.9, 2
    # Shadow
    cv2.putText(f, text, (11, 31), font, scale, (0, 0, 0), thickness + 1)
    # Text
    cv2.putText(f, text, (10, 30), font, scale, (255, 255, 255), thickness)
    return f


def make_comparison(
    original_path: str,
    edited_path: str,
    alpha_path: str | None,
    output_path: str,
) -> None:
    orig_frames, fps, ow, oh = read_video_frames(original_path)
    edit_frames, _, ew, eh = read_video_frames(edited_path)

    # Use original video dimensions as canonical size
    target_w, target_h = ow, oh

    # Align frame counts (use shortest)
    n_frames = min(len(orig_frames), len(edit_frames))
    orig_frames = orig_frames[:n_frames]
    edit_frames = resize_frames(edit_frames[:n_frames], target_w, target_h)

    # Load alpha if provided
    alpha_frames = None
    if alpha_path and Path(alpha_path).exists():
        af, _, aw, ah = read_video_frames(alpha_path)
        alpha_frames = resize_frames(af[:n_frames], target_w, target_h)

    panels = 3 if alpha_frames else 2
    out_w = target_w * panels
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, target_h),
    )

    for i in range(n_frames):
        f_orig = add_label(orig_frames[i], "ORIGINAL")
        f_edit = add_label(edit_frames[i], "EDITED")

        if alpha_frames is not None:
            # Create heat-map overlay of alpha on original
            af = alpha_frames[i]
            if len(af.shape) == 3:
                af_gray = cv2.cvtColor(af, cv2.COLOR_BGR2GRAY)
            else:
                af_gray = af
            af_colour = cv2.applyColorMap(af_gray, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_frames[i], 0.55, af_colour, 0.45, 0)
            overlay = add_label(overlay, "ALPHA MASK")
            row = np.hstack([f_orig, overlay, f_edit])
        else:
            row = np.hstack([f_orig, f_edit])

        writer.write(row)

    writer.release()
    print(f"Comparison video saved: {output_path}")
    print(f"  {n_frames} frames, {fps:.1f} fps, {out_w}×{target_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison video")
    parser.add_argument("--original", required=True, help="Original video path")
    parser.add_argument("--edited", required=True, help="Edited video path")
    parser.add_argument("--alpha", default=None, help="Alpha mask video path")
    parser.add_argument(
        "--output", default="data/output/comparison.mp4",
        help="Output comparison video path"
    )
    args = parser.parse_args()
    make_comparison(args.original, args.edited, args.alpha, args.output)

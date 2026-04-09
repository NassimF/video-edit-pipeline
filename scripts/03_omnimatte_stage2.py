"""
Stage 3: Run Generative Omnimatte Stage 2 (RGBA optimization).

Takes the Casper outputs from Stage 2 and the original sequence directory,
then calls inference/reconstruct_omnimatte.py to produce per-object RGBA layers
with a soft alpha mask capturing both the object and its physical effects
(shadows, reflections, etc.).

Usage:
    conda run -n omnimatte python scripts/03_omnimatte_stage2.py \\
        --object_idx 0 \\
        --num_objects 1

    # For multiple objects:
    conda run -n omnimatte python scripts/03_omnimatte_stage2.py \\
        --object_idx 0 --num_objects 2
    conda run -n omnimatte python scripts/03_omnimatte_stage2.py \\
        --object_idx 1 --num_objects 2

Reads from (set by Stage 2):
    data/output/casper_workspace/pipeline_seq/   ← original sequence
    data/output/casper_outputs/<seq>-fg=*.mp4    ← Casper output videos

Outputs:
    data/output/03_rgba_layers/pipeline_seq/
        fg00_alpha.mp4        ← soft alpha mask video (key output for Stage 4)
        fg00_rgba_checker.mp4 ← RGBA with checker background
        fg00_rgba_constant.mp4
        fg00_visualization.mp4
        bg.mp4
    data/output/03_alpha_masks/alpha_object_0.mp4  ← copy for Stage 4
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

OMNIMATTE_REPO = "/workspace/storage_nassim/gen-omnimatte-public"
BASE_DIR = "/workspace/storage_nassim/Video-restyle"
SEQ_NAME = "pipeline_seq"


def run_omnimatte_optimization(
    data_rootdir: Path,
    source_video_dir: Path,
    save_path: Path,
    run_seqs: str,
    device: str = "cuda:0",
    num_steps: int = 6000,
    sample_size: str = "480x832",
) -> None:
    """Call reconstruct_omnimatte.py via subprocess."""
    save_path.mkdir(parents=True, exist_ok=True)

    script = os.path.join(OMNIMATTE_REPO, "inference", "reconstruct_omnimatte.py")
    cmd = [
        sys.executable, script,
        f"--config.data.data_rootdir={data_rootdir}",
        f"--config.omnimatte.source_video_dir={source_video_dir}",
        f"--config.experiment.run_seqs={run_seqs}",
        f"--config.experiment.save_path={save_path}",
        f"--config.omnimatte.num_steps={num_steps}",
        f"--config.data.sample_size={sample_size}",
        "--config.experiment.skip_if_exists=False",
        f"--config.system.device={device}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    print(f"Running Omnimatte optimization...")
    print(f"  source_video_dir: {source_video_dir}")
    print(f"  save_path:        {save_path}")
    result = subprocess.run(cmd, cwd=OMNIMATTE_REPO, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Omnimatte optimization failed with return code {result.returncode}"
        )


def copy_alpha_for_stage4(rgba_layer_dir: Path, alpha_masks_dir: Path) -> None:
    """
    Copy fg{idx:02d}_alpha.mp4 files to data/output/03_alpha_masks/ for easy
    access by Stage 4 (WAN VACE).

    gen-omnimatte saves alpha videos as:
        {save_path}/{seq_name}/fg{fg_id:02d}_alpha.mp4
    where fg_id = max(0, original_fg_id), so fg00_alpha.mp4, fg01_alpha.mp4, etc.
    """
    alpha_masks_dir.mkdir(parents=True, exist_ok=True)
    seq_out = rgba_layer_dir / SEQ_NAME

    alpha_videos = sorted(seq_out.glob("fg*_alpha.mp4")) if seq_out.exists() else []
    if not alpha_videos:
        print(f"  WARNING: No fg*_alpha.mp4 found under {seq_out}")
        return

    for src in alpha_videos:
        # fg00_alpha.mp4 → alpha_object_0.mp4
        fg_str = src.stem.replace("_alpha", "").replace("fg", "")
        try:
            obj_idx = int(fg_str)
        except ValueError:
            obj_idx = 0
        dst = alpha_masks_dir / f"alpha_object_{obj_idx}.mp4"
        shutil.copy2(src, dst)
        print(f"  Alpha mask → {dst}")


def main(args):
    workspace = Path(BASE_DIR) / "data" / "output" / "casper_workspace"
    casper_outputs = Path(BASE_DIR) / "data" / "output" / "casper_outputs"
    rgba_dir = Path(BASE_DIR) / "data" / "output" / "03_rgba_layers"
    alpha_dir = Path(BASE_DIR) / "data" / "output" / "03_alpha_masks"

    # Validate prerequisite directories exist
    seq_dir = workspace / SEQ_NAME
    if not seq_dir.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: {seq_dir}\n"
            "Run 02_omnimatte_stage1.py first."
        )
    if not casper_outputs.exists():
        raise FileNotFoundError(
            f"Casper outputs directory not found: {casper_outputs}\n"
            "Run 02_omnimatte_stage1.py first."
        )

    # Check that expected Casper output files exist
    fg_pattern = f"{SEQ_NAME}-fg="
    casper_mp4s = [f for f in casper_outputs.glob("*.mp4")
                   if fg_pattern in f.name and "_tuple" not in f.name]
    if not casper_mp4s:
        raise FileNotFoundError(
            f"No Casper output MP4s matching '{fg_pattern}*.mp4' found in {casper_outputs}"
        )
    print(f"Found {len(casper_mp4s)} Casper output file(s):")
    for f in casper_mp4s:
        print(f"  {f.name}")

    # Run optimization
    run_omnimatte_optimization(
        data_rootdir=workspace,
        source_video_dir=casper_outputs,
        save_path=rgba_dir,
        run_seqs=SEQ_NAME,
        device=args.device,
        num_steps=args.num_steps,
        sample_size=args.sample_size,
    )

    # Copy alpha masks for Stage 4
    print("\nCopying alpha masks for Stage 4...")
    copy_alpha_for_stage4(rgba_dir, alpha_dir)

    print("\nOmnimatte Stage 2 complete.")
    print(f"  RGBA layers: {rgba_dir}/{SEQ_NAME}/")
    print(f"  Alpha masks: {alpha_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnimatte Stage 2 (RGBA optimization)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num_steps", type=int, default=6000,
        help="Optimization iterations (paper default: 6000; full quality: 20000)"
    )
    parser.add_argument(
        "--sample_size", default="480x832",
        help="HxW for processing (must match Stage 2)"
    )
    main(parser.parse_args())

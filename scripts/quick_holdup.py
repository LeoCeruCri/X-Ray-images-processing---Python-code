"""Quick check: time-averaged holdup image for each camera of one measurement.

Usage (from the repository root, with the environment active):

    python scripts/quick_holdup.py configs/local/my_run.yaml

Writes ``<output_root>/<measurement>/camera N/mean_holdup.tif`` and a PNG
overview per camera. Uses the uncorrected two-point method (no scatter or
beam-hardening correction yet).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xrayfb import io
from xrayfb.config import load_config
from xrayfb.corrections import subtract_dark
from xrayfb.holdup import two_point_holdup

# Display range of the overview figure (holdup is dimensionless).
PLOT_VMIN, PLOT_VMAX = -0.1, 1.1


def process_camera(cfg, camera: int) -> np.ndarray:
    ref_frames = cfg.reference_frames.to_range()
    dark = io.mean_image(io.camera_folder(cfg.dark, camera), ref_frames)
    empty = subtract_dark(io.mean_image(io.camera_folder(cfg.empty, camera), ref_frames), dark)
    packed = subtract_dark(io.mean_image(io.camera_folder(cfg.packed, camera), ref_frames), dark)
    meas = subtract_dark(
        io.mean_image(io.camera_folder(cfg.measurement, camera), cfg.frames.to_range()), dark
    )
    # Clip at 1 count so that the logarithm is defined; flagged pixels are rare
    # and appear as outliers in the figure.
    return two_point_holdup(np.maximum(meas, 1), np.maximum(empty, 1), np.maximum(packed, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = cfg.output_root / cfg.measurement.relative_to(cfg.raw_root)
    for camera in cfg.cameras:
        holdup = process_camera(cfg, camera)
        cam_dir = out_dir / f"camera {camera}"
        io.write_image(holdup, cam_dir / "mean_holdup.tif", cfg.raw_root)

        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
        im = ax.imshow(holdup, vmin=PLOT_VMIN, vmax=PLOT_VMAX, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Holdup (-)")
        ax.set_title(f"Camera {camera}: time-averaged holdup")
        fig.savefig(cam_dir / "mean_holdup.png", dpi=150)
        plt.close(fig)
        print(f"Camera {camera}: median holdup {np.nanmedian(holdup):.3f} -> {cam_dir}")


if __name__ == "__main__":
    main()

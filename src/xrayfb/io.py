"""Reading and writing of TIFF images.

Folder layout produced by the TU Delft acquisition software::

    <measurement>/camera 1/img_0.tif, img_1.tif, ...
    <measurement>/camera 2/...
    <measurement>/camera 3/...

Rules enforced here:
- Raw data are only ever read. Images are returned as new float arrays.
- Writing is refused anywhere inside the raw-data root.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

FRAME_PATTERN = re.compile(r"^img_(\d+)\.tiff?$", re.IGNORECASE)
CAMERA_FOLDER = "camera {camera}"
WORK_DTYPE = np.float64


def camera_folder(measurement: Path, camera: int) -> Path:
    """Return the folder of one camera inside a measurement folder."""
    folder = Path(measurement) / CAMERA_FOLDER.format(camera=camera)
    if not folder.is_dir():
        raise FileNotFoundError(f"Camera folder not found: {folder}")
    return folder


def list_frames(folder: Path) -> dict[int, Path]:
    """Map frame number -> file for every img_<n>.tif in ``folder``, sorted by frame."""
    frames = {}
    for path in Path(folder).iterdir():
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames[int(match.group(1))] = path
    if not frames:
        raise FileNotFoundError(f"No img_<n>.tif files in {folder}")
    return dict(sorted(frames.items()))


def read_frame(path: Path) -> np.ndarray:
    """Read one TIFF as a float array, checking the stored integer range.

    The detector values are converted to float without any intermediate cast
    to a narrower integer type, so no value can wrap around.
    """
    raw = tifffile.imread(path)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D image in {path}, got shape {raw.shape}")
    return raw.astype(WORK_DTYPE)


def read_stack(folder: Path, frames: range | list[int]) -> np.ndarray:
    """Read the requested frames of one camera as an array (n_frames, rows, cols)."""
    available = list_frames(folder)
    missing = [f for f in frames if f not in available]
    if missing:
        raise KeyError(f"Frames not found in {folder}: {missing[:10]}")
    return np.stack([read_frame(available[f]) for f in frames])


def mean_image(folder: Path, frames: range | list[int]) -> np.ndarray:
    """Time-averaged image of one camera, computed frame by frame to limit memory use."""
    available = list_frames(folder)
    total = None
    for f in frames:
        img = read_frame(available[f])
        total = img if total is None else total + img
    if total is None:
        raise ValueError("Empty frame selection")
    return total / len(frames)


def _is_inside(path: Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def write_image(image: np.ndarray, path: Path, raw_root: Path) -> None:
    """Write a float32 TIFF, refusing any destination inside ``raw_root``."""
    if _is_inside(path, raw_root):
        raise PermissionError(f"Refusing to write inside the raw-data folder: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(image, dtype=np.float32))

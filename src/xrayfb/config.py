"""Loading of run configuration files (YAML).

Every processing parameter lives in a YAML file under ``configs/``; the
code holds no hard-coded experimental values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FrameRange:
    start: int
    stop: int
    step: int = 1

    def to_range(self) -> range:
        return range(self.start, self.stop, self.step)


@dataclass(frozen=True)
class RunConfig:
    raw_root: Path
    output_root: Path
    cameras: tuple[int, ...]
    measurement: Path
    dark: Path
    empty: Path
    packed: Path
    frames: FrameRange
    reference_frames: FrameRange


def load_config(path: Path) -> RunConfig:
    """Read a run configuration and check that the referenced folders exist."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    raw_root = Path(cfg["raw_root"])
    output_root = Path(cfg["output_root"])
    if output_root.resolve().is_relative_to(raw_root.resolve()):
        raise ValueError("output_root must not be inside raw_root")

    def raw(key: str) -> Path:
        folder = raw_root / cfg["folders"][key]
        if not folder.is_dir():
            raise FileNotFoundError(f"{key}: folder not found: {folder}")
        return folder

    return RunConfig(
        raw_root=raw_root,
        output_root=output_root,
        cameras=tuple(cfg["cameras"]),
        measurement=raw("measurement"),
        dark=raw("dark"),
        empty=raw("empty"),
        packed=raw("packed"),
        frames=FrameRange(**cfg["frames"]),
        reference_frames=FrameRange(**cfg["reference_frames"]),
    )

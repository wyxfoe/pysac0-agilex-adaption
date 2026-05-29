"""Inspect the contents of an AgileX / RoboTwin HDF5 episode.

Prints:
  * file-level attributes (e.g. ``sim``, ``compress``)
  * every group and dataset (shape, dtype, attrs)
  * per-dimension stats (min / max / mean / std) for numeric datasets
  * optional sample frames (``--show N``) for arrays
  * image datasets are summarized (size + dtype), not dumped

Examples
--------
Quick overview of a single episode::

    python inspect_hdf5.py /home/wyx/pysac0-agilex_adapt/episode_0.hdf5

Compare a few episodes in a directory::

    python inspect_hdf5.py /home/wyx/data/pick_place --num_episodes 3

Dump the first/last 3 frames of every numeric dataset::

    python inspect_hdf5.py episode_0.hdf5 --show 3

Focus on a single dataset::

    python inspect_hdf5.py episode_0.hdf5 --only /action --show 5
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import h5py
import numpy as np


# Datasets whose shape suggests image tensors — skip numeric stats (too large).
IMAGE_HINTS = ("images", "rgb", "image_raw", "depth")


def is_image_like(name: str, dataset: h5py.Dataset) -> bool:
    lower = name.lower()
    if any(h in lower for h in IMAGE_HINTS):
        return True
    # Shape-based heuristic: (T, H, W, C) or (T, H, W) with H,W >= 64.
    if dataset.ndim in (3, 4) and dataset.shape[-2] >= 64 and dataset.shape[-3] >= 64:
        return True
    return False


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:7.2f} TB"


def print_attrs(obj: h5py.HLObject, prefix: str) -> None:
    if len(obj.attrs) == 0:
        return
    for key, value in obj.attrs.items():
        print(f"{prefix}  @{key} = {value!r}")


def summarize_dataset(
    name: str,
    dataset: h5py.Dataset,
    show: int = 0,
    max_cols: int = 16,
) -> None:
    kind = "image" if is_image_like(name, dataset) else "numeric"
    nbytes = np.prod(dataset.shape) * dataset.dtype.itemsize
    comp = dataset.compression or "raw"
    print(
        f"  [{kind:7s}] {name:<42s} "
        f"shape={tuple(dataset.shape)}  dtype={dataset.dtype}  "
        f"size={format_bytes(int(nbytes))}  compression={comp}"
    )
    print_attrs(dataset, prefix="   ")

    if kind == "image":
        # Only show a single-pixel sample so users can sanity-check BGR/RGB range.
        if dataset.ndim >= 3 and dataset.shape[0] > 0:
            try:
                sample = dataset[0]
                if sample.ndim == 1:  # compressed (bytes) frame
                    print(f"     first frame bytes head: {bytes(sample[:16])!r}... "
                          f"(likely JPEG/PNG encoded, len={len(sample)})")
                else:
                    print(f"     first frame min={sample.min()}, max={sample.max()}, "
                          f"mean={sample.mean():.2f}, dtype={sample.dtype}")
            except Exception as exc:
                print(f"     (cannot read first frame: {exc})")
        return

    # Numeric dataset → per-dimension stats.
    try:
        arr = dataset[()]
    except Exception as exc:  # e.g. vlen or object dtype
        print(f"     (skip stats: {exc})")
        return

    if arr.dtype.kind not in "fiub":  # float/int/unsigned/bool
        print(f"     (skip stats: non-numeric dtype {arr.dtype})")
        return

    arr = np.asarray(arr)
    if arr.ndim == 1:
        print(f"     flat stats: min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  n={arr.shape[0]}")
    else:
        # Assume leading axis = time.  Show per-dim stats if trailing dim is small.
        trailing = arr.shape[-1]
        if trailing <= max_cols:
            flat = arr.reshape(-1, trailing)
            mins = flat.min(axis=0)
            maxs = flat.max(axis=0)
            means = flat.mean(axis=0)
            stds = flat.std(axis=0)
            header = "       " + "  ".join(f"d{i:02d}" for i in range(trailing))
            print(header)

            def _row(label: str, vec: np.ndarray) -> None:
                print(f"     {label:4s} " + "  ".join(f"{v:+6.3f}" for v in vec))

            _row("min", mins)
            _row("max", maxs)
            _row("mean", means)
            _row("std", stds)
        else:
            flat = arr.reshape(-1)
            print(f"     global stats: min={flat.min():.4f} max={flat.max():.4f} "
                  f"mean={flat.mean():.4f} std={flat.std():.4f}")

    if show > 0 and arr.shape[0] > 0:
        n = min(show, arr.shape[0])
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        print(f"     first {n}:")
        for i in range(n):
            print(f"       [{i:3d}] {arr[i]}")
        if arr.shape[0] > n:
            print(f"     last  {n}:")
            for i in range(arr.shape[0] - n, arr.shape[0]):
                print(f"       [{i:3d}] {arr[i]}")


def walk(
    obj: h5py.Group,
    only: Optional[str],
    show: int,
    max_cols: int,
) -> None:
    """Recursively print every group/dataset under ``obj``."""
    def visitor(name: str, node: h5py.HLObject) -> None:
        full = "/" + name
        if only and not full.startswith(only):
            # We still need to descend into parents of `only`. h5py visititems
            # visits depth-first so children of non-matching groups will still
            # arrive; a prefix mismatch just means this exact node is outside
            # the filter, but we can't easily short-circuit descent, so check
            # prefix both ways.
            if not only.startswith(full):
                return
        if isinstance(node, h5py.Group):
            print(f"\n[group] {full}")
            print_attrs(node, prefix="  ")
        elif isinstance(node, h5py.Dataset):
            if only and not full.startswith(only):
                return
            summarize_dataset(full, node, show=show, max_cols=max_cols)
        else:
            print(f"[other] {full}  type={type(node).__name__}")

    # Root attributes first.
    print("[root] /")
    print_attrs(obj, prefix="  ")
    obj.visititems(visitor)


def inspect_file(
    path: str,
    only: Optional[str],
    show: int,
    max_cols: int,
) -> None:
    print("=" * 80)
    print(f"File: {path}")
    size = os.path.getsize(path)
    print(f"Size: {format_bytes(size)}")
    print("=" * 80)
    with h5py.File(path, "r") as f:
        walk(f, only=only, show=show, max_cols=max_cols)


def gather_files(target: str, num_episodes: Optional[int]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "episode_*.hdf5")))
        if not files:
            files = sorted(glob.glob(os.path.join(target, "*.hdf5")))
        if num_episodes is not None:
            files = files[:num_episodes]
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in {target}")
        return files
    raise FileNotFoundError(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect every topic / dataset in an HDF5 episode.",
    )
    parser.add_argument("target", type=str,
                        help="Path to an .hdf5 file or a directory containing episode_*.hdf5")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="If TARGET is a directory, process only the first N files.")
    parser.add_argument("--only", type=str, default=None,
                        help="Restrict output to datasets whose full path starts with this prefix, "
                             "e.g. --only /observations/qpos or --only /action")
    parser.add_argument("--show", type=int, default=0,
                        help="Print the first and last N rows of every numeric dataset (default 0).")
    parser.add_argument("--max_cols", type=int, default=16,
                        help="Per-dimension stats are printed only when the trailing dim <= this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for f in gather_files(args.target, args.num_episodes):
        inspect_file(f, only=args.only, show=args.show, max_cols=args.max_cols)


if __name__ == "__main__":
    main()

"""
Extract GEN1 dataset from 7z archives and convert raw .td.dat files to .td.dat.h5 (HDF5).

This script:
1. Extracts 7z archives containing the Prophesee GEN1 dataset
2. Converts raw .td.dat event files to .td.dat.h5 format expected by preprocess_dataset.py
3. Copies .npy label files as-is
4. Organizes everything into {test, train, val}/ directory structure

Usage:
    python extract_and_convert_gen1.py /path/to/7z/archives /path/to/output/gen1

The output directory will have:
    gen1/
    ├── test/
    │   ├── ..._bbox.npy
    │   ├── ..._td.dat.h5
    ├── train/
    │   ├── ..._bbox.npy
    │   ├── ..._td.dat.h5
    └── val/
        ├── ..._bbox.npy
        └── ..._td.dat.h5
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import py7zr
from tqdm import tqdm

# Add project root so we can import Prophesee tools
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.evaluation.prophesee.io.dat_events_tools import parse_header, EV_TYPE

# GEN1 sensor dimensions
GEN1_HEIGHT = 240
GEN1_WIDTH = 304


def convert_dat_file_to_h5(dat_path: Path, h5_path: Path):
    """
    Convert a raw Prophesee .dat event file to HDF5 format.

    The output .h5 file will have the structure:
        events/
            x: uint16
            y: uint16
            p: uint8 (polarity, 0 or 1)
            t: uint32 (timestamp in microseconds)
            height: scalar int
            width: scalar int
    """
    with open(dat_path, "rb") as f:
        _, ev_type, ev_size, size = parse_header(f)
        dat = np.fromfile(f, dtype=EV_TYPE)

    if len(dat) == 0:
        print(f"  WARNING: Empty .dat file, skipping: {dat_path.name}")
        return False

    # Decode bitfields: x (bits 0-13), y (bits 14-27), p (bit 28)
    x = np.bitwise_and(dat["_"], 16383).astype(np.uint16)
    y = np.right_shift(np.bitwise_and(dat["_"], 268419072), 14).astype(np.uint16)
    p = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28).astype(np.uint8)
    t = dat["t"]  # already uint32

    with h5py.File(str(h5_path), "w") as h5f:
        events = h5f.create_group("events")
        events.create_dataset("x", data=x)
        events.create_dataset("y", data=y)
        events.create_dataset("p", data=p)
        events.create_dataset("t", data=t)
        events.create_dataset("height", data=np.array(GEN1_HEIGHT))
        events.create_dataset("width", data=np.array(GEN1_WIDTH))

    return True


def process_archive(archive_path: Path, output_dir: Path, dry_run: bool = False):
    """Extract a single 7z archive and convert .dat files to .h5."""
    print(f"\nProcessing: {archive_path.name}")

    with py7zr.SevenZipFile(str(archive_path), "r") as z:
        names = z.getnames()

    # Count files
    dat_files = [n for n in names if n.endswith("_td.dat")]
    npy_files = [n for n in names if n.endswith("_bbox.npy")]
    print(f"  Found {len(dat_files)} .dat files, {len(npy_files)} .npy files")

    if dry_run:
        for n in names[:5]:
            print(f"    {n}")
        return

    # Check which files still need to be processed
    files_to_extract = []
    for name in names:
        parts = Path(name).parts
        split = None
        for part in parts:
            if part in ("test", "train", "val"):
                split = part
                break
        if split is None:
            continue

        filename = parts[-1]
        split_dir = output_dir / split

        if filename.endswith("_td.dat"):
            h5_path = split_dir / (filename + ".h5")
            if not h5_path.exists():
                files_to_extract.append(name)
        elif filename.endswith(".npy"):
            npy_path = split_dir / filename
            if not npy_path.exists():
                files_to_extract.append(name)

    if not files_to_extract:
        print("  All files already processed, skipping.")
        return

    print(f"  {len(files_to_extract)} files to extract/convert...")

    # Extract to a temp directory, process, then clean up
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  Extracting archive to temp dir...")
        with py7zr.SevenZipFile(str(archive_path), "r") as z:
            z.extractall(path=tmp_dir)

        # Process extracted files
        for name in tqdm(files_to_extract, desc=f"  {archive_path.stem}"):
            extracted_path = Path(tmp_dir) / name
            if not extracted_path.exists():
                print(f"  WARNING: Expected file not found: {name}")
                continue

            parts = Path(name).parts
            split = None
            for part in parts:
                if part in ("test", "train", "val"):
                    split = part
                    break

            filename = parts[-1]
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            if filename.endswith("_td.dat"):
                h5_path = split_dir / (filename + ".h5")
                convert_dat_file_to_h5(extracted_path, h5_path)
            elif filename.endswith(".npy"):
                npy_path = split_dir / filename
                shutil.copy2(str(extracted_path), str(npy_path))

    print(f"  Done with {archive_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract GEN1 7z archives and convert .dat to .h5"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing the 7z archives",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory (will create test/train/val subdirs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list contents without extracting",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    archives = sorted(source_dir.glob("*.7z"))
    if not archives:
        print(f"No .7z files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(archives)} archives:")
    for a in archives:
        size_gb = a.stat().st_size / (1024 ** 3)
        print(f"  {a.name} ({size_gb:.1f} GB)")

    print(f"\nOutput directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        process_archive(archive, output_dir, dry_run=args.dry_run)

    # Print summary
    print("\n=== Summary ===")
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            h5_count = len(list(split_dir.glob("*.h5")))
            npy_count = len(list(split_dir.glob("*.npy")))
            print(f"  {split}/: {h5_count} .h5 files, {npy_count} .npy files")


if __name__ == "__main__":
    main()

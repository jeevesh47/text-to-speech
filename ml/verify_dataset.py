"""
Dataset verification script.

Validates all recorded ``.npy`` files under the ``dataset/`` directory:

* Lists every sign class and sample count.
* Verifies each file's shape matches ``(SEQUENCE_LENGTH, TOTAL_FEATURES)``.
* Reports per-class statistics (non-zero frame ratio, feature ranges).
* Breaks down counts by subject/participant ID.
* Flags corrupted or wrong-shape files.
* Identifies individual samples with low hand-detection coverage.

This script does **not** require a webcam — it only reads saved files.

Usage
─────
    python -m ml.verify_dataset
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

from ml.config import (
    DATASET_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    FEATURES_PER_HAND,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Minimum percentage of frames that must contain at least one hand.
# Samples below this threshold are reported for review.
MIN_HAND_COVERAGE = 85.0


def _parse_subject(stem: str) -> str:
    """Extract subject ID from a filename stem like 's01_sample_003'."""
    parts = stem.split("_sample_")
    return parts[0] if len(parts) == 2 else "unknown"


def verify_dataset(dataset_dir: Path | None = None) -> None:
    """Run verification and print a human-readable report."""
    ds_dir = dataset_dir or DATASET_DIR

    if not ds_dir.exists():
        print(f"  Dataset directory not found: {ds_dir}")
        print("  Record samples first:  python -m ml.collect_data")
        return

    sign_dirs = sorted(
        d for d in ds_dir.iterdir()
        if d.is_dir() and d.name != ".gitkeep"
    )

    if not sign_dirs:
        print("  No sign classes found in dataset directory.")
        print("  Record samples first:  python -m ml.collect_data")
        return

    expected_shape = (SEQUENCE_LENGTH, TOTAL_FEATURES)

    print("=" * 70)
    print("  ISL Dataset Verification Report")
    print("=" * 70)
    print(f"  Dataset directory : {ds_dir}")
    print(f"  Expected shape    : {expected_shape}")
    print(f"  Min hand coverage: {MIN_HAND_COVERAGE:.0f}%")
    print()

    grand_total = 0
    grand_errors = 0
    all_subjects: set[str] = set()

    for sign_dir in sign_dirs:
        sign_name = sign_dir.name
        samples = sorted(sign_dir.glob("*.npy"))

        if not samples:
            print(f"  [{sign_name}]  —  empty (no .npy files)")
            print()
            continue

        valid_count = 0
        error_count = 0
        valid_data: list[np.ndarray] = []
        subjects_here: set[str] = set()
        valid_by_subject: dict[str, int] = {}

        for path in samples:
            try:
                data = np.load(path)
            except Exception as exc:
                print(f"    [FAIL] {path.name}: CORRUPT - {exc}")
                error_count += 1
                continue

            subj = _parse_subject(path.stem)
            subjects_here.add(subj)
            all_subjects.add(subj)

            if data.shape != expected_shape:
                print(
                    f"    [FAIL] {path.name}: shape {data.shape} "
                    f"!= expected {expected_shape}"
                )
                error_count += 1
                continue

            valid_count += 1
            valid_data.append(data)
            valid_by_subject[subj] = (
                valid_by_subject.get(subj, 0) + 1
            )

            # --------------------------------------------------------
            # Individual sample hand coverage
            # --------------------------------------------------------

            left_present = np.any(
                np.abs(data[:, :FEATURES_PER_HAND]) > 1e-6,
                axis=1,
            )

            right_present = np.any(
                np.abs(data[:, FEATURES_PER_HAND:]) > 1e-6,
                axis=1,
            )

            any_hand = left_present | right_present

            hand_coverage = float(any_hand.mean() * 100.0)
            left_coverage = float(left_present.mean() * 100.0)
            right_coverage = float(right_present.mean() * 100.0)

            if hand_coverage < MIN_HAND_COVERAGE:
                print(
                    f"    [LOW COVERAGE] {path.name}: "
                    f"any-hand={hand_coverage:.1f}%, "
                    f"left={left_coverage:.1f}%, "
                    f"right={right_coverage:.1f}%"
                )

        grand_total += valid_count
        grand_errors += error_count

        # ── Per-class summary ──
        print(f"  [{sign_name}]")
        print(
            f"    Samples  : {valid_count} valid, "
            f"{error_count} error(s)"
        )
        print(f"    Subjects : {sorted(subjects_here)}")

        # Per-subject breakdown
        for subj in sorted(subjects_here):
            print(
                f"      {subj} : "
                f"{valid_by_subject.get(subj, 0)} sample(s)"
            )

        if valid_data:
            stacked = np.concatenate(
                valid_data,
                axis=0,
            )  # (N*seq_len, 126)

            total_frames = stacked.shape[0]

            # Left / right hand presence
            left_nonzero = np.any(
                stacked[:, :FEATURES_PER_HAND] != 0,
                axis=1,
            ).sum()

            right_nonzero = np.any(
                stacked[:, FEATURES_PER_HAND:] != 0,
                axis=1,
            ).sum()

            any_nonzero = np.any(
                stacked != 0,
                axis=1,
            ).sum()

            print(
                f"    Frames with any hand   : "
                f"{any_nonzero}/{total_frames} "
                f"({any_nonzero / total_frames:.0%})"
            )

            print(
                f"    Frames with left hand  : "
                f"{left_nonzero}/{total_frames}"
            )

            print(
                f"    Frames with right hand : "
                f"{right_nonzero}/{total_frames}"
            )

            # Feature range
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)

            print(
                f"    Feature mean range : "
                f"[{mean.min():.4f}, {mean.max():.4f}]"
            )

            print(
                f"    Feature std  range : "
                f"[{std.min():.4f}, {std.max():.4f}]"
            )

        print()

    # ── Grand summary ──
    print("─" * 70)
    print(f"  Classes  : {len(sign_dirs)}")
    print(
        f"  Samples  : {grand_total} valid, "
        f"{grand_errors} error(s)"
    )
    print(f"  Subjects : {sorted(all_subjects)}")
    print("─" * 70)

    if grand_errors:
        print()
        print(
            "  [!] Some files have errors - "
            "re-record or delete them."
        )

    if grand_total == 0:
        print()
        print("  No valid samples found.  Record data with:")
        print("    python -m ml.collect_data")

    print()


if __name__ == "__main__":
    verify_dataset()
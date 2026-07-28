#!/usr/bin/env python3
"""Verify saved tensors and operational SNR labels in a prepared dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch


def calculate_operational_snr(tensor: torch.Tensor) -> float:
    tensor = tensor.to(torch.float32)
    mean = float(tensor.mean())
    standard_deviation = float(tensor.std(unbiased=False))
    return 20.0 * math.log10(mean / standard_deviation)


def verify(dataset_root: Path, tolerance: float) -> None:
    labels_path = dataset_root / "labels.csv"
    labels = pd.read_csv(labels_path)
    required = {"split", "noisy_image", "SNR_classical_dB", "noise_variance"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing required columns in {labels_path}: {sorted(missing)}")

    failures: list[str] = []
    for row in labels.itertuples(index=False):
        path = dataset_root / row.noisy_image
        if not path.is_file():
            failures.append(f"missing file: {path}")
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            failures.append(f"unexpected shape {tuple(tensor.shape)}: {path}")
            continue
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            failures.append(f"invalid tensor range or non-finite values: {path}")
            continue
        recomputed = calculate_operational_snr(tensor)
        if abs(recomputed - float(row.SNR_classical_dB)) > tolerance:
            failures.append(
                f"label mismatch {recomputed:.8f} vs {row.SNR_classical_dB}: {path}"
            )

    expected_per_reference = labels["noise_variance"].nunique()
    if "source_image" in labels.columns:
        variants = labels.groupby("source_image").size()
        if not (variants == expected_per_reference).all():
            failures.append("one or more references do not have every noise variance")
        leakage = labels.groupby("source_image")["split"].nunique()
        if not (leakage == 1).all():
            failures.append("reference-image leakage was detected across splits")

    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"Verification failed with {len(failures)} issue(s):\n{preview}")
    print(
        f"Verified {len(labels)} samples, {labels['noise_variance'].nunique()} "
        f"noise variances, and no reference-image leakage."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    verify(args.dataset_root, args.tolerance)


if __name__ == "__main__":
    main()

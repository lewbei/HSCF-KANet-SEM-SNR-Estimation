#!/usr/bin/env python3
"""Prepare the controlled Gaussian-noise datasets used in the manuscript.

Clean reference images are split before noise generation so every noisy
variant of one reference remains in exactly one split. Ten zero-mean Gaussian
noise variances (0.001--0.010) are used by default. The supervised target is
the operational image-level amplitude-ratio SNR defined in the manuscript:

    20 * log10(mean(corrupted_image) / std(corrupted_image))
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_VARIANCES = tuple(i / 1000 for i in range(1, 11))


def load_grayscale(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def operational_snr_db(corrupted: torch.Tensor) -> float:
    mean = float(corrupted.mean())
    standard_deviation = float(corrupted.std(unbiased=False))
    if mean <= 0.0 or standard_deviation <= 0.0:
        raise ValueError("The corrupted image must have positive mean and variance")
    return 20.0 * math.log10(mean / standard_deviation)


def split_references(
    references: list[Path], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[Path]]:
    shuffled = references.copy()
    random.Random(seed).shuffle(shuffled)
    n_train = int(train_ratio * len(shuffled))
    n_val = int(val_ratio * len(shuffled))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_dataset(
    images_root: Path,
    output_root: Path,
    dataset_name: str,
    variances: tuple[float, ...] = DEFAULT_VARIANCES,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    size: int = 256,
) -> None:
    references = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not references:
        raise RuntimeError(f"No supported images were found under {images_root}")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("Train and validation ratios must be between zero and one")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train and validation ratios must sum to less than one")
    if any(variance <= 0.0 for variance in variances):
        raise ValueError("Noise variances must be positive")

    splits = split_references(references, train_ratio, val_ratio, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    reference_index = {path: index for index, path in enumerate(references)}
    for split, split_references_list in splits.items():
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        split_rows: list[dict[str, object]] = []

        for reference in tqdm(split_references_list, desc=f"Generating {split}"):
            clean = load_grayscale(reference, size)
            relative = reference.relative_to(images_root)
            safe_stem = relative.with_suffix("").as_posix().replace("/", "__")
            clean_index = reference_index[reference]

            for variance_index, variance in enumerate(variances):
                generator = torch.Generator().manual_seed(
                    seed + clean_index * len(variances) + variance_index
                )
                noise = torch.randn(clean.shape, generator=generator) * math.sqrt(variance)
                corrupted = (clean + noise).clamp(0.0, 1.0).to(torch.float32)
                snr_db = operational_snr_db(corrupted)
                filename = f"{safe_stem}_variance_{variance:.3f}.pt"
                torch.save(corrupted, split_root / filename)

                split_row = {
                    "filename": filename,
                    "snr_db": snr_db,
                    "split": split,
                    "source_image": relative.as_posix(),
                    "noise_variance": variance,
                }
                split_rows.append(split_row)
                all_rows.append(
                    {
                        **split_row,
                        "dataset": dataset_name,
                        "noisy_image": f"{split}/{filename}",
                        "SNR_classical_dB": snr_db,
                    }
                )

        pd.DataFrame(split_rows).to_csv(split_root / "labels.csv", index=False)

    pd.DataFrame(all_rows).to_csv(output_root / "labels.csv", index=False)
    counts = {name: len(items) * len(variances) for name, items in splits.items()}
    print(f"Prepared {sum(counts.values())} samples in {output_root}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Clean-image directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset directory, for example Dataset/Biofilms",
    )
    parser.add_argument("--dataset-name", required=True, help="Biofilms, EPFL, or NFFA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--variances",
        type=float,
        nargs="+",
        default=DEFAULT_VARIANCES,
        help="Gaussian noise variances (default: 0.001 through 0.010)",
    )
    args = parser.parse_args()
    prepare_dataset(
        images_root=args.images,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        variances=tuple(args.variances),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        size=args.size,
    )


if __name__ == "__main__":
    main()

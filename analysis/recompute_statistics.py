#!/usr/bin/env python3
"""Recompute the canonical summary statistics from per-seed result rows.

The inferential units are five seed-matched runs. Paired tests are performed
separately for each in-domain dataset or cross-dataset transfer direction.
Holm correction is applied across competitors within each metric family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ("mae", "rmse", "r2")
PROPOSED = "proposed"


def confidence_interval(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * sd / np.sqrt(n)
    return mean, sd, mean - margin, mean + margin, t_crit


def summarise_groups(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group_values, group in frame.groupby(keys, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row: dict[str, float | int | str] = dict(zip(keys, group_values))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            mean, sd, low, high, _ = confidence_interval(group[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_lo"] = low
            row[f"{metric}_ci95_hi"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    source: str,
    test: str | None,
) -> list[dict[str, float | int | str]]:
    proposed = frame[frame["architecture"] == PROPOSED].set_index("seed")
    competitors = sorted(set(frame["architecture"]) - {PROPOSED})
    rows: list[dict[str, float | int | str]] = []

    for metric in METRICS:
        family_rows: list[dict[str, float | int | str]] = []
        for competitor in competitors:
            other = frame[frame["architecture"] == competitor].set_index("seed")
            common = sorted(set(proposed.index).intersection(other.index))
            if len(common) < 2:
                continue

            proposed_values = proposed.loc[common, metric].to_numpy(dtype=float)
            competitor_values = other.loc[common, metric].to_numpy(dtype=float)
            difference = proposed_values - competitor_values
            mean_diff, _, diff_low, diff_high, _ = confidence_interval(difference)
            t_result = stats.ttest_rel(proposed_values, competitor_values)
            try:
                wilcoxon_p = float(
                    stats.wilcoxon(
                        proposed_values,
                        competitor_values,
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
            except ValueError:
                wilcoxon_p = 1.0

            family_rows.append(
                {
                    "scope": scope,
                    "source_dataset": source,
                    "test_dataset": "" if test is None else test,
                    "metric": metric,
                    "competitor": competitor,
                    "n_pairs": len(common),
                    "proposed_mean": float(np.mean(proposed_values)),
                    "competitor_mean": float(np.mean(competitor_values)),
                    "mean_diff": mean_diff,
                    "diff_ci95_lo": diff_low,
                    "diff_ci95_hi": diff_high,
                    # Preserved for compatibility with the canonical v2 file:
                    # this field counts proposed values below the competitor.
                    "proposed_wins": int(np.sum(proposed_values < competitor_values)),
                    "t_stat": float(t_result.statistic),
                    "p_paired_t": float(t_result.pvalue),
                    "p_wilcoxon": wilcoxon_p,
                }
            )

        paired_t_adjusted = holm_adjust([float(row["p_paired_t"]) for row in family_rows])
        wilcoxon_adjusted = holm_adjust([float(row["p_wilcoxon"]) for row in family_rows])
        for row, paired_p, wilcoxon_p in zip(
            family_rows, paired_t_adjusted, wilcoxon_adjusted
        ):
            row["p_paired_t_holm"] = paired_p
            row["p_wilcoxon_holm"] = wilcoxon_p
        rows.extend(family_rows)

    return rows


def recompute(input_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_csv)
    required = {
        "architecture",
        "source_dataset",
        "test_dataset",
        "eval_type",
        "seed",
        *METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_key = ["architecture", "source_dataset", "test_dataset", "seed"]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("Duplicate architecture/source/test/seed rows were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    in_domain = frame[frame["source_dataset"] == frame["test_dataset"]].copy()
    cross_domain = frame[frame["source_dataset"] != frame["test_dataset"]].copy()

    in_summary = summarise_groups(in_domain, ["architecture", "source_dataset"])
    in_summary.to_csv(output_dir / "STATS_indomain_mean_sd_ci.csv", index=False)

    cross_summary = summarise_groups(
        cross_domain, ["architecture", "source_dataset", "test_dataset"]
    )
    cross_summary.to_csv(output_dir / "STATS_crossdataset_mean_sd_ci.csv", index=False)

    pooled_rows: list[dict[str, float | int | str]] = []
    for architecture, group in in_domain.groupby("architecture", sort=True):
        n_runs = len(group)
        row: dict[str, float | int | str] = {
            "architecture": architecture,
            "n_runs": n_runs,
            "df": n_runs - 1,
        }
        t_crit = float(stats.t.ppf(0.975, df=n_runs - 1))
        row["t_crit"] = t_crit
        for metric in METRICS:
            mean, sd, low, high, _ = confidence_interval(group[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_lo"] = low
            row[f"{metric}_ci95_hi"] = high
        row["note"] = (
            "Pooled over 3 datasets x 5 seeds = 15 runs; CI uses t(14)=2.145 "
            "with sqrt(15). Treat as descriptive: seeds repeat across datasets, "
            "so per-dataset CIs (STATS_indomain_mean_sd_ci.csv, n=5, "
            "t(4)=2.776) are the canonical inferential units."
        )
        pooled_rows.append(row)
    pd.DataFrame(pooled_rows).to_csv(
        output_dir / "STATS_indomain_overall_mean_sd_ci.csv", index=False
    )

    tests: list[dict[str, float | int | str]] = []
    for dataset in sorted(in_domain["source_dataset"].unique()):
        subset = in_domain[in_domain["source_dataset"] == dataset]
        tests.extend(
            paired_rows(
                subset,
                scope="in-domain per dataset (5 seed pairs)",
                source=dataset,
                test=None,
            )
        )

    directions = (
        cross_domain[["source_dataset", "test_dataset"]]
        .drop_duplicates()
        .sort_values(["source_dataset", "test_dataset"])
    )
    for direction in directions.itertuples(index=False):
        subset = cross_domain[
            (cross_domain["source_dataset"] == direction.source_dataset)
            & (cross_domain["test_dataset"] == direction.test_dataset)
        ]
        tests.extend(
            paired_rows(
                subset,
                scope="cross-dataset per direction (5 seed pairs)",
                source=direction.source_dataset,
                test=direction.test_dataset,
            )
        )

    test_columns = [
        "scope",
        "source_dataset",
        "test_dataset",
        "metric",
        "competitor",
        "n_pairs",
        "proposed_mean",
        "competitor_mean",
        "mean_diff",
        "diff_ci95_lo",
        "diff_ci95_hi",
        "proposed_wins",
        "t_stat",
        "p_paired_t",
        "p_paired_t_holm",
        "p_wilcoxon",
        "p_wilcoxon_holm",
    ]
    tests_frame = pd.DataFrame(tests)[test_columns]
    scope_order = {
        "in-domain per dataset (5 seed pairs)": 0,
        "cross-dataset per direction (5 seed pairs)": 1,
    }
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    tests_frame = (
        tests_frame.assign(
            _scope_order=tests_frame["scope"].map(scope_order),
            _metric_order=tests_frame["metric"].map(metric_order),
        )
        .sort_values(
            [
                "_scope_order",
                "_metric_order",
                "source_dataset",
                "test_dataset",
                "competitor",
            ],
            kind="stable",
        )
        .drop(columns=["_scope_order", "_metric_order"])
    )
    tests_frame.to_csv(output_dir / "PAIRED_TESTS_proposed_vs_others.csv", index=False)

    print(f"Wrote canonical-format statistics to {output_dir}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root
        / "results"
        / "per_seed"
        / "ALL_MODELS_PER_SEED_RESULTS.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "recomputed_statistics",
    )
    args = parser.parse_args()
    recompute(args.input, args.output_dir)


if __name__ == "__main__":
    main()

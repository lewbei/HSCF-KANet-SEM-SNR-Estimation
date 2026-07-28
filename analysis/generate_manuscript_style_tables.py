#!/usr/bin/env python3
"""Generate v2 table fragments while preserving modified.tex table formatting.

The manuscript is read only as a formatting template.  It is never edited.
Existing rows are refreshed from the canonical v2 CSVs, and ARNIQA-SNR plus
TOPIQ-NR-SNR are inserted into the applicable comparison tables.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(os.environ.get("HSCF_MANUSCRIPT", ROOT / "manuscript" / "modified.tex"))
DATA = ROOT / "results"
OUT = ROOT / "generated_tables"

INDOMAIN_CSV = DATA / "statistics" / "STATS_indomain_mean_sd_ci.csv"
CROSS_CSV = DATA / "statistics" / "STATS_crossdataset_mean_sd_ci.csv"
PAIRED_CSV = DATA / "statistics" / "PAIRED_TESTS_proposed_vs_others.csv"
TIMING_CSV = DATA / "metadata" / "INFERENCE_BENCHMARK.csv"
FAILURE_CSV = DATA / "predictions" / "HSCFKANET_FAILURE_CASES.csv"
SOTA_COMPLEXITY_CSV = DATA / "summaries" / "sota_complexity_summary.csv"

TABLES = [
    ("01_in_domain_epfl.tex", "tab:all_performance_EPFL", "replace"),
    ("02_ablation_epfl.tex", "tab:abltation_EPFL", "replace"),
    ("03_cross_dataset_train_epfl.tex", "tab:cross_epfl_test_nffa_biofilm", "replace"),
    ("04_in_domain_nffa.tex", "tab:all_performance_NFFA", "replace"),
    ("05_ablation_nffa.tex", "tab:ablation_nffa", "replace"),
    ("06_cross_dataset_train_nffa.tex", "tab:cross_nffa_test_epfl_biofilm", "replace"),
    ("07_in_domain_biofilms.tex", "tab:all_performance_biofilm", "replace"),
    ("08_ablation_biofilms.tex", "tab:ablation_biofilm", "replace"),
    ("09_cross_dataset_train_biofilms.tex", "tab:cross_biofilm_test_epfl_nffa", "replace"),
    ("10_model_complexity.tex", "tab:model_complexity", "replace"),
    ("11_paired_tests_in_domain.tex", "tab:t_test_results", "replace"),
    ("12_failure_cases.tex", "tab:hscf_failure_cases", "insert"),
]

FULL_LABEL_TO_ARCH = {
    r"ResNet18 \citep{He_Zhang_Ren_Sun_2015}": "ResNet18Regression",
    r"ResNeXt50 \citep{xie_aggregated_2017}": "ResNeXt50Regression",
    r"ConvNeXt Tiny \citep{9879745}": "ConvNextTinyRegression",
    r"ViT \citep{dosovitskiy2021vit}": "ViTRegression",
    r"CalibNet \citep{Lew_Sim_Tan_2025}": "CalibNet",
    "HSCF-KANet": "proposed",
}

SOTA_FULL_LABELS = {
    "ARNIQA_SNR": r"ARNIQA-SNR \citep{agnolucci2024arniqa}",
    "TOPIQ_NR_SNR": r"TOPIQ-NR-SNR \citep{chen2024topiq}",
}

ABLATION_LABEL_TO_ARCH = {
    "CNN + MLP": "CNN_MLP",
    "CNN + KAN": "CNN_KAN",
    "CNN + KAN + PSD": "CNN_KAN_PSD",
    "CNN + KAN + Local": "CNN_KAN_Local",
    "CNN + RF": "CNN_RF",
    "HSCF-KANet": "proposed",
}

CROSS_LABEL_TO_ARCH = {
    "ResNet18": "ResNet18Regression",
    "ResNeXt50": "ResNeXt50Regression",
    "ConvNext Tiny": "ConvNextTinyRegression",
    "ViT": "ViTRegression",
    "CNN + MLP": "CNN_MLP",
    "CNN + RF": "CNN_RF",
    "CalibNet": "CalibNet",
    "CNN + KAN": "CNN_KAN",
    "CNN + KAN + PSD": "CNN_KAN_PSD",
    "CNN + KAN + Local": "CNN_KAN_Local",
    "HSCF-KANet": "proposed",
}

PAIR_LABEL_TO_ARCH = {
    "CNN + MLP": "CNN_MLP",
    "CNN + KAN": "CNN_KAN",
    "CNN + KAN + PSD": "CNN_KAN_PSD",
    "CNN + KAN + Local": "CNN_KAN_Local",
    "CNN + RF": "CNN_RF",
    "ResNet18": "ResNet18Regression",
    "ResNeXt50": "ResNeXt50Regression",
}

DATASET_FROM_TARGET = {"EPFL": "EPFL", "NFFA": "NFFA", "Biofilm": "Biofilms"}
DISPLAY_DATASET = {"EPFL": "EPFL", "NFFA": "NFFA", "Biofilms": "Biofilm"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_table(manuscript: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    label_at = manuscript.index(marker)
    start = manuscript.rfind(r"\begin{table", 0, label_at)
    if start < 0:
        raise ValueError(f"No table start found for {label}")
    end_marker = r"\end{table}"
    end = manuscript.index(end_marker, label_at) + len(end_marker)
    return manuscript[start:end]


def leading_space(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def row_ending(line: str) -> str:
    pos = line.rfind(r"\\")
    if pos < 0:
        raise ValueError(f"No LaTeX row ending found: {line}")
    return line[pos:]


def pm(row: dict[str, str], metric: str) -> str:
    return f"${float(row[f'{metric}_mean']):.4f} \\pm {float(row[f'{metric}_sd']):.4f}$"


def metric_row(indent: str, label: str, row: dict[str, str], ending: str) -> str:
    return (
        f"{indent}{label} & {pm(row, 'rmse')} & {pm(row, 'r2')} & "
        f"{pm(row, 'mae')} {ending}"
    )


def refresh_full_table(
    block: str,
    dataset: str,
    indomain: dict[tuple[str, str], dict[str, str]],
) -> str:
    output: list[str] = []
    inserted = False
    replaced = set()
    for line in block.splitlines():
        first = line.split("&", 1)[0].strip()
        arch = FULL_LABEL_TO_ARCH.get(first)
        if arch is None:
            output.append(line)
            continue
        ending = row_ending(line)
        indent = leading_space(line)
        if arch == "proposed":
            for sota_arch in ("ARNIQA_SNR", "TOPIQ_NR_SNR"):
                output.append(
                    metric_row(
                        indent,
                        SOTA_FULL_LABELS[sota_arch],
                        indomain[(dataset, sota_arch)],
                        ending,
                    )
                )
            inserted = True
        output.append(metric_row(indent, first, indomain[(dataset, arch)], ending))
        replaced.add(arch)
    assert replaced == set(FULL_LABEL_TO_ARCH.values())
    assert inserted
    return "\n".join(output)


def refresh_ablation_table(
    block: str,
    dataset: str,
    indomain: dict[tuple[str, str], dict[str, str]],
) -> str:
    output: list[str] = []
    replaced = set()
    for line in block.splitlines():
        first = line.split("&", 1)[0].strip()
        arch = ABLATION_LABEL_TO_ARCH.get(first)
        if arch is None:
            output.append(line)
            continue
        output.append(
            metric_row(
                leading_space(line), first, indomain[(dataset, arch)], row_ending(line)
            )
        )
        replaced.add(arch)
    assert replaced == set(ABLATION_LABEL_TO_ARCH.values())
    rendered = "\n".join(output)
    assert "ARNIQA" not in rendered and "TOPIQ" not in rendered
    return rendered


def refresh_cross_table(
    block: str,
    source: str,
    cross: dict[tuple[str, str, str], dict[str, str]],
) -> str:
    output: list[str] = []
    inserted_targets: set[str] = set()
    replaced = 0
    for line in block.splitlines():
        parts = line.split("&")
        if len(parts) < 5:
            output.append(line)
            continue
        model_label = parts[0].strip()
        target_label = parts[1].strip()
        arch = CROSS_LABEL_TO_ARCH.get(model_label)
        target = DATASET_FROM_TARGET.get(target_label)
        if arch is None or target is None:
            output.append(line)
            continue
        indent = leading_space(line)
        ending = row_ending(line)
        output.append(
            metric_row(indent, f"{model_label} & {target_label}", cross[(source, target, arch)], ending)
        )
        replaced += 1
        if arch == "CalibNet":
            for sota_arch, sota_label in (
                ("ARNIQA_SNR", "ARNIQA-SNR"),
                ("TOPIQ_NR_SNR", "TOPIQ-NR-SNR"),
            ):
                output.append(
                    metric_row(
                        indent,
                        f"{sota_label} & {target_label}",
                        cross[(source, target, sota_arch)],
                        ending,
                    )
                )
            inserted_targets.add(target)
    assert replaced == 22, f"Expected 22 original cross rows; replaced {replaced}"
    assert len(inserted_targets) == 2
    return "\n".join(output)


def add_sota_complexity_rows(
    block: str,
    timings: list[dict[str, str]],
    sota_complexity: list[dict[str, str]],
) -> str:
    timing_by_arch = {
        row["architecture"]: row
        for row in timings
        if int(row["batch_size"]) == 1
    }
    complexity_by_key = {
        (row["model"], row["source_dataset"]): row for row in sota_complexity
    }

    new_rows: list[str] = []
    for arch, model, citation in (
        ("ARNIQA_SNR", "ARNIQA-SNR", "agnolucci2024arniqa"),
        ("TOPIQ_NR_SNR", "TOPIQ-NR-SNR", "chen2024topiq"),
    ):
        epfl = complexity_by_key[(model, "EPFL")]
        nffa = complexity_by_key[(model, "NFFA")]
        biofilms = complexity_by_key[(model, "Biofilms")]
        params_m = int(epfl["total_params"]) / 1_000_000
        gflops = float(epfl["gflops_256x256"])
        inference = float(timing_by_arch[arch]["median_ms_per_sample"])
        new_rows.append(
            f"{model} \\citep{{{citation}}} & {params_m:.2f} & {gflops:.3f} & "
            f"{inference:.2f} & {float(biofilms['training_time_min_mean']):.1f} & "
            f"{float(epfl['training_time_min_mean']):.1f} & "
            f"{float(nffa['training_time_min_mean']):.1f} \\\\ \\hline"
        )

    output: list[str] = []
    inserted = False
    for line in block.splitlines():
        output.append(line)
        if line.strip().startswith(r"CalibNet \citep{Lew_Sim_Tan_2025} &"):
            output.extend(new_rows)
            inserted = True
    assert inserted
    return "\n".join(output)


def paired_row(
    dataset: str,
    label: str,
    metric: str,
    row: dict[str, str],
    indent: str,
    ending: str,
) -> str:
    p_value = float(row["p_paired_t_holm"])
    significant = "Yes" if p_value < 0.05 else "No"
    return (
        f"{indent}{dataset} & {label} & {metric.upper()} & "
        f"${float(row['t_stat']):.2f}$ & {p_value:.3f} & {significant} {ending}"
    )


def refresh_paired_table(block: str, paired: list[dict[str, str]]) -> str:
    selected = [row for row in paired if row["scope"].startswith("in-domain")]
    by_key = {
        (row["source_dataset"], row["competitor"], row["metric"]): row
        for row in selected
    }
    assert len(selected) == 108

    last_existing = {
        ("EPFL", "CNN + RF", "MAE"),
        ("NFFA", "CNN + RF", "MAE"),
        ("Biofilm", "ResNeXt50", "MAE"),
    }
    output: list[str] = []
    inserted_datasets: set[str] = set()
    replaced = 0
    for line in block.splitlines():
        parts = [part.strip() for part in line.split("&")]
        if len(parts) < 6 or parts[0] not in DATASET_FROM_TARGET:
            output.append(line)
            continue
        dataset_label, comparison, metric = parts[:3]
        arch = PAIR_LABEL_TO_ARCH.get(comparison)
        if arch is None or metric.lower() not in ("rmse", "mae"):
            output.append(line)
            continue
        dataset = DATASET_FROM_TARGET[dataset_label]
        indent = leading_space(line)
        ending = row_ending(line)
        output.append(
            paired_row(
                dataset_label,
                comparison,
                metric,
                by_key[(dataset, arch, metric.lower())],
                indent,
                ending,
            )
        )
        replaced += 1
        if (dataset_label, comparison, metric) in last_existing:
            for sota_arch, sota_label in (
                ("ARNIQA_SNR", "ARNIQA-SNR"),
                ("TOPIQ_NR_SNR", "TOPIQ-NR-SNR"),
            ):
                for sota_metric in ("rmse", "mae"):
                    output.append(
                        paired_row(
                            dataset_label,
                            sota_label,
                            sota_metric,
                            by_key[(dataset, sota_arch, sota_metric)],
                            indent,
                            ending,
                        )
                    )
            inserted_datasets.add(dataset)
    assert replaced == 34
    assert len(inserted_datasets) == 3
    # Keep the visible manuscript layout/header unchanged; record the corrected
    # p-value definition in a non-rendered comment for the person replacing it.
    return (
        "% The p column contains Holm-adjusted paired-t p-values from canonical v2.\n"
        + "\n".join(output)
    )


def make_failure_table(rows: list[dict[str, str]]) -> str:
    selections = [
        ("Biofilms", "NP11ii1_png_clean_var001.png", "BF-1"),
        ("Biofilms", "NP11i2_png_clean_var004.png", "BF-2"),
        ("EPFL", "frame_236_png_clean_var002.png", "EPFL-1"),
        ("EPFL", "frame_72_png_clean_var009.png", "EPFL-2"),
        ("NFFA", "L7_ac0f8dc1ae7aa0f9ddbf943fd70df999_jpg_clean_var001.png", "NFFA-1"),
        ("NFFA", "L7_cd3ebf2d244508c38312dddf46af0e48_jpg_clean_var001.png", "NFFA-2"),
    ]
    by_key = {(row["test_dataset"], row["image_id"]): row for row in rows}
    lines = ["% Case-to-image mapping:"]
    for dataset, image_id, case in selections:
        assert (dataset, image_id) in by_key
        lines.append(f"% {case}: {image_id}")
    lines.extend(
        [
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Representative in-domain failure cases of HSCF-KANet with target SNR, predicted SNR, signed error, and absolute error}",
            r"\label{tab:hscf_failure_cases}",
            r"\small",
            r"\begin{tabular}{|l|l|c|c|c|c|}\hline",
            r"Dataset & Case & Target SNR (dB) & Predicted SNR (dB) & Signed Error (dB) & Absolute Error (dB) \\ \hline",
        ]
    )
    for dataset, image_id, case in selections:
        row = by_key[(dataset, image_id)]
        lines.append(
            f"{DISPLAY_DATASET[dataset]} & {case} & {float(row['target_snr_db']):.4f} & "
            f"{float(row['predicted_snr_db']):.4f} & {float(row['signed_error_db']):.4f} & "
            f"{float(row['absolute_error_db']):.4f} \\\\ \\hline"
        )
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def write(name: str, contents: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(contents.rstrip() + "\n", encoding="utf-8")


def make_readme() -> str:
    lines = [
        "# Manuscript-style canonical v2 table replacements",
        "",
        "Every replacement table was copied from the corresponding table block in",
        "`v2/SCv1/modified.tex`. Its environment, column specification, header, caption,",
        "label, model order, borders, spacing, and styling are retained. Existing learned-model",
        "values were refreshed from the canonical v2 results, and ARNIQA-SNR plus",
        "TOPIQ-NR-SNR were inserted where applicable.",
        "",
        "The manuscript and its bibliography were not modified.",
        "",
        "| File | Label | Action |",
        "|---|---|---|",
    ]
    for filename, label, action in TABLES:
        lines.append(f"| `{filename}` | `{label}` | {action} |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The three ablation tables contain only the six existing ablation models; no SOTA rows were added.",
            "- The other replacement tables retain the row composition already present in the manuscript and add the two SOTA models.",
            "- Table 11 uses Holm-adjusted paired-t p-values from the canonical v2 package while retaining the manuscript's six-column layout.",
            "- Add the entries in `sota_iqa_bibliography_entries.bib` to `reference.bib` when you perform the manual replacement.",
            "- File 12 is a new failure-case table and therefore has no pre-existing manuscript block.",
        ]
    )
    return "\n".join(lines)


def make_preview() -> str:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.45in]{geometry}",
        r"\usepackage{array,graphicx,amsmath}",
        r"\providecommand{\citep}[1]{[#1]}",
        r"\begin{document}",
    ]
    for filename, _, _ in TABLES:
        lines.extend([rf"\input{{{filename}}}", r"\clearpage"])
    lines.append(r"\end{document}")
    return "\n".join(lines)


def validate_outputs(original_blocks: dict[str, str]) -> None:
    for filename, label, action in TABLES:
        rendered = (OUT / filename).read_text(encoding="utf-8")
        assert rendered.count(r"\begin{table") == 1
        assert rendered.count(r"\end{table}") == 1
        assert rendered.count(rf"\label{{{label}}}") == 1
        assert rendered.count(r"\begin{tabular}") == 1
        assert rendered.count(r"\end{tabular}") == 1
        if action == "replace":
            original = original_blocks[label]
            original_caption = re.search(r"\\caption\{.*\}", original).group(0)
            original_begin = original.splitlines()[0]
            original_tabular = next(
                line for line in original.splitlines() if r"\begin{tabular}" in line
            )
            assert original_caption in rendered
            assert original_begin in rendered
            assert original_tabular in rendered

    for filename in ("02_ablation_epfl.tex", "05_ablation_nffa.tex", "08_ablation_biofilms.tex"):
        rendered = (OUT / filename).read_text(encoding="utf-8")
        assert "ARNIQA" not in rendered and "TOPIQ" not in rendered
    for filename in (
        "01_in_domain_epfl.tex",
        "04_in_domain_nffa.tex",
        "07_in_domain_biofilms.tex",
        "10_model_complexity.tex",
    ):
        rendered = (OUT / filename).read_text(encoding="utf-8")
        assert rendered.count("ARNIQA-SNR") == 1
        assert rendered.count("TOPIQ-NR-SNR") == 1
    for filename in (
        "03_cross_dataset_train_epfl.tex",
        "06_cross_dataset_train_nffa.tex",
        "09_cross_dataset_train_biofilms.tex",
    ):
        rendered = (OUT / filename).read_text(encoding="utf-8")
        assert rendered.count("ARNIQA-SNR") == 2
        assert rendered.count("TOPIQ-NR-SNR") == 2


def main() -> None:
    manuscript_hash_before = sha256(MANUSCRIPT)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    original_blocks = {
        label: extract_table(manuscript, label)
        for _, label, action in TABLES
        if action == "replace"
    }

    indomain_rows = read_csv(INDOMAIN_CSV)
    cross_rows = read_csv(CROSS_CSV)
    paired_rows = read_csv(PAIRED_CSV)
    timing_rows = read_csv(TIMING_CSV)
    failure_rows = read_csv(FAILURE_CSV)
    sota_complexity_rows = read_csv(SOTA_COMPLEXITY_CSV)
    assert len(indomain_rows) == 39
    assert len(cross_rows) == 78
    assert len(paired_rows) == 324

    indomain = {
        (row["source_dataset"], row["architecture"]): row for row in indomain_rows
    }
    cross = {
        (row["source_dataset"], row["test_dataset"], row["architecture"]): row
        for row in cross_rows
    }

    write(
        "01_in_domain_epfl.tex",
        refresh_full_table(original_blocks["tab:all_performance_EPFL"], "EPFL", indomain),
    )
    write(
        "02_ablation_epfl.tex",
        refresh_ablation_table(original_blocks["tab:abltation_EPFL"], "EPFL", indomain),
    )
    write(
        "03_cross_dataset_train_epfl.tex",
        refresh_cross_table(original_blocks["tab:cross_epfl_test_nffa_biofilm"], "EPFL", cross),
    )
    write(
        "04_in_domain_nffa.tex",
        refresh_full_table(original_blocks["tab:all_performance_NFFA"], "NFFA", indomain),
    )
    write(
        "05_ablation_nffa.tex",
        refresh_ablation_table(original_blocks["tab:ablation_nffa"], "NFFA", indomain),
    )
    write(
        "06_cross_dataset_train_nffa.tex",
        refresh_cross_table(original_blocks["tab:cross_nffa_test_epfl_biofilm"], "NFFA", cross),
    )
    write(
        "07_in_domain_biofilms.tex",
        refresh_full_table(original_blocks["tab:all_performance_biofilm"], "Biofilms", indomain),
    )
    write(
        "08_ablation_biofilms.tex",
        refresh_ablation_table(original_blocks["tab:ablation_biofilm"], "Biofilms", indomain),
    )
    write(
        "09_cross_dataset_train_biofilms.tex",
        refresh_cross_table(original_blocks["tab:cross_biofilm_test_epfl_nffa"], "Biofilms", cross),
    )
    write(
        "10_model_complexity.tex",
        add_sota_complexity_rows(
            original_blocks["tab:model_complexity"], timing_rows, sota_complexity_rows
        ),
    )
    write(
        "11_paired_tests_in_domain.tex",
        refresh_paired_table(original_blocks["tab:t_test_results"], paired_rows),
    )
    write("12_failure_cases.tex", make_failure_table(failure_rows))
    write("README.md", make_readme())
    write("_preview_all_tables.tex", make_preview())
    write(
        "sota_iqa_bibliography_entries.bib",
        r"""@inproceedings{agnolucci2024arniqa,
  title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={189--198},
  year={2024}
}

@article{chen2024topiq,
  author={Chen, Chaofeng and Mo, Jiadi and Hou, Jingwen and Wu, Haoning and Liao, Liang and Sun, Wenxiu and Yan, Qiong and Lin, Weisi},
  title={TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  volume={33},
  pages={2404--2418},
  doi={10.1109/TIP.2024.3378466}
}""",
    )

    validate_outputs(original_blocks)
    assert sha256(MANUSCRIPT) == manuscript_hash_before, "Manuscript was unexpectedly modified"
    print(f"Generated and validated {len(TABLES)} manuscript-style table fragments in {OUT}")


if __name__ == "__main__":
    main()

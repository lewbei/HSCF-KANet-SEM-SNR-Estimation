#!/usr/bin/env python3
"""Generate standalone LaTeX replacement tables from the canonical v2 results.

This script deliberately does not edit the manuscript.  It writes one complete
table environment per file under v2/SCv1/replacement_tables_v2/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
OUT = ROOT / "generated_tables"

INDOMAIN_CSV = DATA / "statistics" / "STATS_indomain_mean_sd_ci.csv"
CROSS_CSV = DATA / "statistics" / "STATS_crossdataset_mean_sd_ci.csv"
PAIRED_CSV = DATA / "statistics" / "PAIRED_TESTS_proposed_vs_others.csv"
TIMING_CSV = DATA / "metadata" / "INFERENCE_BENCHMARK.csv"
CHECKPOINT_CSV = DATA / "metadata" / "CHECKPOINT_HASHES.csv"
FAILURE_CSV = DATA / "predictions" / "HSCFKANET_FAILURE_CASES.csv"

SEEDS = (42, 123, 456, 789, 1024)

FULL_MODEL_ORDER = [
    "ResNet18Regression",
    "ResNeXt50Regression",
    "ConvNextTinyRegression",
    "ViTRegression",
    "CalibNet",
    "CNN_MLP",
    "CNN_KAN",
    "CNN_KAN_PSD",
    "CNN_KAN_Local",
    "CNN_RF",
    "ARNIQA_SNR",
    "TOPIQ_NR_SNR",
    "proposed",
]

# General performance/cross-dataset tables show the external or standard
# comparison models only. Component/head variants are reserved for ablation
# tables so the same rows are not duplicated across table types.
COMPARISON_MODEL_ORDER = [
    "ResNet18Regression",
    "ResNeXt50Regression",
    "ConvNextTinyRegression",
    "ViTRegression",
    "CalibNet",
    "ARNIQA_SNR",
    "TOPIQ_NR_SNR",
    "proposed",
]

ABLATION_ORDER = [
    "CNN_MLP",
    "CNN_KAN",
    "CNN_KAN_PSD",
    "CNN_KAN_Local",
    "CNN_RF",
    "proposed",
]

PAIR_ORDER = [model for model in FULL_MODEL_ORDER if model != "proposed"]

MODEL_LABEL = {
    "ResNet18Regression": r"ResNet18 \citep{He_Zhang_Ren_Sun_2015}",
    "ResNeXt50Regression": r"ResNeXt50 \citep{xie_aggregated_2017}",
    "ConvNextTinyRegression": r"ConvNeXt Tiny \citep{9879745}",
    "ViTRegression": r"ViT \citep{dosovitskiy2021vit}",
    "CalibNet": r"CalibNet \citep{Lew_Sim_Tan_2025}",
    "CNN_MLP": "CNN + MLP",
    "CNN_KAN": "CNN + KAN",
    "CNN_KAN_PSD": "CNN + KAN + PSD",
    "CNN_KAN_Local": "CNN + KAN + Local",
    "CNN_RF": "CNN + RF",
    "ARNIQA_SNR": r"ARNIQA-SNR \citep{agnolucci2024arniqa}",
    "TOPIQ_NR_SNR": r"TOPIQ-NR-SNR \citep{chen2024topiq}",
    "proposed": "HSCF-KANet",
}

MODEL_SHORT = {
    model: label.split(r" \citep")[0] for model, label in MODEL_LABEL.items()
}

DATASET_DISPLAY = {
    "EPFL": "EPFL CVLab",
    "NFFA": "NFFA-EUROPE",
    "Biofilms": "Biofilm",
}

CLASSICAL = {
    "EPFL": [
        (r"Nearest Neighbour Interpolation \citep{sim_image_2015}", 20.514, -45.093, 20.422),
        (r"Linear Interpolation \citep{lew_single_2025}", 14.852, -23.161, 14.828),
        (r"Combined Nearest + Linear Method \citep{lew_single_2025}", 18.223, -35.373, 18.164),
        (r"Basic Cubic Spline Interpolation \citep{Sim_Ting_Leong_Tso_2019}", 20.514, -45.093, 20.422),
        (r"Nonlinear Least Squares Regression \citep{Sim_Norhisham_2016}", 17.646, -33.107, 17.597),
        (r"ACLDR \citep{Sim_Lim_Yeap_2016}", 22.208, -53.021, 22.002),
        (r"ASNN \citep{sim_image_2015}", 0.693, 0.947, 0.515),
        (r"CSILLSR \citep{Sim_Ting_Leong_Tso_2019}", 0.738, 0.940, 0.569),
        (r"QSE \citep{lew_single_2025}", 22.174, -52.850, 21.972),
    ],
    "NFFA": [
        (r"Nearest Neighbour Interpolation \citep{sim_image_2015}", 16.379, -19.345, 16.014),
        (r"Linear Interpolation \citep{lew_single_2025}", 14.886, -15.805, 14.327),
        (r"Combined Nearest + Linear Method \citep{lew_single_2025}", 15.658, -17.594, 15.274),
        (r"Basic Cubic Spline Interpolation \citep{Sim_Ting_Leong_Tso_2019}", 16.379, -19.345, 16.014),
        (r"Nonlinear Least Squares Regression \citep{Sim_Norhisham_2016}", 15.472, -17.155, 14.985),
        (r"ACLDR \citep{Sim_Lim_Yeap_2016}", 17.086, -21.139, 16.726),
        (r"ASNN \citep{sim_image_2015}", 3.342, 0.153, 2.633),
        (r"CSILLSR \citep{Sim_Ting_Leong_Tso_2019}", 3.484, 0.080, 2.838),
        (r"QSE \citep{lew_single_2025}", 16.963, -20.821, 16.595),
    ],
    "Biofilms": [
        (r"Nearest Neighbour Interpolation \citep{sim_image_2015}", 11.202, -5.201, 10.588),
        (r"Linear Interpolation \citep{lew_single_2025}", 10.596, -4.548, 9.773),
        (r"Combined Nearest + Linear Method \citep{lew_single_2025}", 10.846, -4.812, 10.206),
        (r"Basic Cubic Spline Interpolation \citep{Sim_Ting_Leong_Tso_2019}", 11.202, -5.201, 10.588),
        (r"Nonlinear Least Squares Regression \citep{Sim_Norhisham_2016}", 10.806, -4.770, 10.039),
        (r"ACLDR \citep{Sim_Lim_Yeap_2016}", 11.787, -5.865, 11.158),
        (r"ASNN \citep{sim_image_2015}", 3.575, 0.369, 2.961),
        (r"CSILLSR \citep{Sim_Ting_Leong_Tso_2019}", 3.872, 0.259, 3.252),
        (r"QSE \citep{lew_single_2025}", 11.638, -5.692, 11.013),
    ],
}

TABLE_MAP = [
    ("01_in_domain_epfl.tex", r"tab:all_performance_EPFL", "replace"),
    ("02_ablation_epfl.tex", r"tab:abltation_EPFL", "replace"),
    ("03_cross_dataset_train_epfl.tex", r"tab:cross_epfl_test_nffa_biofilm", "replace"),
    ("04_in_domain_nffa.tex", r"tab:all_performance_NFFA", "replace"),
    ("05_ablation_nffa.tex", r"tab:ablation_nffa", "replace"),
    ("06_cross_dataset_train_nffa.tex", r"tab:cross_nffa_test_epfl_biofilm", "replace"),
    ("07_in_domain_biofilms.tex", r"tab:all_performance_biofilm", "replace"),
    ("08_ablation_biofilms.tex", r"tab:ablation_biofilm", "replace"),
    ("09_cross_dataset_train_biofilms.tex", r"tab:cross_biofilm_test_epfl_nffa", "replace"),
    ("10_model_complexity.tex", r"tab:model_complexity", "replace"),
    ("11_paired_tests_in_domain.tex", r"tab:t_test_results", "replace"),
    ("12_failure_cases.tex", r"tab:hscf_failure_cases", "insert"),
]

# Captions are reproduced verbatim from v2/SCv1/modified.tex so that replacing a
# table does not alter its list-of-tables entry or surrounding manuscript wording.
ORIGINAL_CAPTIONS = {
    "tab:all_performance_EPFL": "Performance comparison of QSE, CalibNet and HSCF-KANet against interpolation and deep learning methods on EPFL CVLab Test Dataset",
    "tab:abltation_EPFL": "Performance of the HSCF-KANet and Ablation Study on EPFL CVLab Test Dataset",
    "tab:cross_epfl_test_nffa_biofilm": "Cross-Dataset Evaluation of Trained Model on EPFL CVLab dataset against the NFFA-EUROPE Dataset and Biofilm Dataset",
    "tab:all_performance_NFFA": "Performance comparison of QSE, CalibNet and HSCF-KANet against interpolation and deep learning methods on NFFA-EUROPE Test dataset",
    "tab:ablation_nffa": "Performance of the HSCF-KANet and Ablation Study on NFFA-EUROPE Test Dataset",
    "tab:cross_nffa_test_epfl_biofilm": "Cross-Dataset Evaluation of Trained Model on NFFA-EUROPE dataset against the EPFL CVLab Dataset and Biofilm Dataset",
    "tab:all_performance_biofilm": "Performance comparison of QSE, CalibNet and HSCF-KANet against interpolation and deep learning methods on Biofilm Test Dataset",
    "tab:ablation_biofilm": "Performance of the HSCF-KANet and Ablation Study on Biofilm Test Dataset",
    "tab:cross_biofilm_test_epfl_nffa": "Cross-Dataset Evaluation of Trained Model on Biofilm dataset against the EPFL CVLab Dataset and NFFA-EUROPE Dataset",
    "tab:model_complexity": "Comparison of Model Parameters, Complexity, Inference Time, and Training Time",
    "tab:t_test_results": "Two-tailed paired Student's t-test results comparing HSCF-KANet against other deep learning models on in-domain test sets ($n = 5$ runs, $df = 4$, $\\alpha = 0.05$)",
}

EXPECTED_LATEX_ROWS = {
    "01_in_domain_epfl.tex": 20,
    "02_ablation_epfl.tex": 7,
    "03_cross_dataset_train_epfl.tex": 10,
    "04_in_domain_nffa.tex": 20,
    "05_ablation_nffa.tex": 7,
    "06_cross_dataset_train_nffa.tex": 10,
    "07_in_domain_biofilms.tex": 20,
    "08_ablation_biofilms.tex": 7,
    "09_cross_dataset_train_biofilms.tex": 10,
    "10_model_complexity.tex": 14,
    "11_paired_tests_in_domain.tex": 37,
    "12_failure_cases.tex": 8,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def pm(mean: float, sd: float, bold: bool = False) -> str:
    body = f"{mean:.4f} \\pm {sd:.4f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def classical_number(value: float) -> str:
    return f"${value:.3f}$"


def original_caption(label: str) -> str:
    return rf"\caption{{{ORIGINAL_CAPTIONS[label]}}}"


def best_model(rows: dict[str, dict[str, str]], metric: str) -> str:
    key = f"{metric}_mean"
    values = {model: float(row[key]) for model, row in rows.items()}
    chooser = max if metric == "r2" else min
    return chooser(values, key=values.get)


def make_full_table(
    dataset: str,
    label: str,
    stats: dict[tuple[str, str], dict[str, str]],
) -> str:
    learned = {model: stats[(dataset, model)] for model in COMPARISON_MODEL_ORDER}
    best = {metric: best_model(learned, metric) for metric in ("rmse", "r2", "mae")}
    display = DATASET_DISPLAY[dataset]
    lines = [
        "% Generated from STATS_indomain_mean_sd_ci.csv (canonical v2).",
        r"\begin{table}[htbp]",
        r"\centering",
        original_caption(label),
        rf"\label{{{label}}}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\begin{tabular}{p{0.55\linewidth}ccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RMSE} & \textbf{$R^2$} & \textbf{MAE} \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Classical estimators}} \\",
    ]
    for model, rmse, r2, mae in CLASSICAL[dataset]:
        lines.append(
            f"{model} & {classical_number(rmse)} & {classical_number(r2)} & "
            f"{classical_number(mae)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Learning-based models}} \\",
        ]
    )
    for model in COMPARISON_MODEL_ORDER:
        row = learned[model]
        cells = []
        for metric in ("rmse", "r2", "mae"):
            cells.append(
                pm(
                    float(row[f"{metric}_mean"]),
                    float(row[f"{metric}_sd"]),
                    model == best[metric],
                )
            )
        lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def make_ablation_table(
    dataset: str,
    label: str,
    stats: dict[tuple[str, str], dict[str, str]],
) -> str:
    rows = {model: stats[(dataset, model)] for model in ABLATION_ORDER}
    best = {metric: best_model(rows, metric) for metric in ("rmse", "r2", "mae")}
    display = DATASET_DISPLAY[dataset]
    lines = [
        "% Generated from STATS_indomain_mean_sd_ci.csv (canonical v2).",
        r"\begin{table}[htbp]",
        r"\centering",
        original_caption(label),
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RMSE} & \textbf{$R^2$} & \textbf{MAE} \\",
        r"\midrule",
    ]
    for model in ABLATION_ORDER:
        row = rows[model]
        cells = [
            pm(
                float(row[f"{metric}_mean"]),
                float(row[f"{metric}_sd"]),
                model == best[metric],
            )
            for metric in ("rmse", "r2", "mae")
        ]
        lines.append(f"{MODEL_SHORT[model]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_cross_table(
    source: str,
    targets: tuple[str, str],
    label: str,
    stats: dict[tuple[str, str, str], dict[str, str]],
) -> str:
    source_display = DATASET_DISPLAY[source]
    target_rows = {
        target: {model: stats[(source, target, model)] for model in COMPARISON_MODEL_ORDER}
        for target in targets
    }
    best = {
        (target, metric): best_model(target_rows[target], metric)
        for target in targets
        for metric in ("rmse", "r2", "mae")
    }
    t1, t2 = targets
    lines = [
        "% Generated from STATS_crossdataset_mean_sd_ci.csv (canonical v2).",
        r"\begin{table}[htbp]",
        r"\centering",
        original_caption(label),
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            rf"\textbf{{Model}} & \multicolumn{{3}}{{c}}{{\textbf{{{DATASET_DISPLAY[t1]}}}}} "
            rf"& \multicolumn{{3}}{{c}}{{\textbf{{{DATASET_DISPLAY[t2]}}}}} \\"
        ),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"& \textbf{RMSE} & \textbf{$R^2$} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$} & \textbf{MAE} \\",
        r"\midrule",
    ]
    for model in COMPARISON_MODEL_ORDER:
        cells = []
        for target in targets:
            row = target_rows[target][model]
            for metric in ("rmse", "r2", "mae"):
                cells.append(
                    pm(
                        float(row[f"{metric}_mean"]),
                        float(row[f"{metric}_sd"]),
                        model == best[(target, metric)],
                    )
                )
        lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def make_complexity_table(
    timings: list[dict[str, str]],
    checkpoints: list[dict[str, str]],
) -> str:
    timing_by_key = {(row["architecture"], int(row["batch_size"])): row for row in timings}
    checkpoint_sizes: dict[str, list[float]] = defaultdict(list)
    for row in checkpoints:
        checkpoint_sizes[row["architecture"]].append(float(row["file_size_mb"]))

    order = [
        "CNN_MLP",
        "CNN_KAN",
        "CNN_KAN_PSD",
        "CNN_KAN_Local",
        "CNN_RF",
        "proposed",
        "CalibNet",
        "ResNet18Regression",
        "ResNeXt50Regression",
        "ConvNextTinyRegression",
        "ARNIQA_SNR",
        "TOPIQ_NR_SNR",
        "ViTRegression",
    ]
    lines = [
        "% Generated only from INFERENCE_BENCHMARK.csv and CHECKPOINT_HASHES.csv.",
        r"\begin{table}[htbp]",
        r"\centering",
        original_caption("tab:model_complexity"),
        r"\label{tab:model_complexity}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Params (M)} & \textbf{Checkpoint (MB)} & "
            r"\textbf{Batch 1 (ms/image)} & \textbf{Batch 32 (ms/image)} \\"
        ),
        r"\midrule",
    ]
    for model in order:
        batch1 = timing_by_key[(model, 1)]
        batch32 = timing_by_key[(model, 32)]
        params_m = int(batch1["total_params"]) / 1_000_000
        sizes = checkpoint_sizes[model]
        if model == "CNN_RF":
            checkpoint = f"{min(sizes):.2f}--{max(sizes):.2f}"
        else:
            checkpoint = f"{sum(sizes) / len(sizes):.2f}"
        lines.append(
            f"{MODEL_LABEL[model]} & {params_m:.3f} & {checkpoint} & "
            f"{float(batch1['median_ms_per_sample']):.3f} & "
            f"{float(batch32['median_ms_per_sample']):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize\textit{Note:} Each timing used 50 warm-up and 200 timed iterations with CUDA synchronization and excluded image I/O. CNN + RF includes the Random Forest prediction step. Its checkpoint size varies by training dataset because the serialized forest is data-dependent; the table therefore reports its observed range. Other checkpoint entries are means over datasets and seeds.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def p_cell(value: float) -> str:
    shown = "<0.0001" if value < 0.0001 else f"{value:.4f}"
    if value < 0.05:
        return rf"$\mathbf{{{shown}}}$"
    return f"${shown}$"


def make_paired_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["scope"].startswith("in-domain")]
    by_key = {
        (row["source_dataset"], row["competitor"], row["metric"]): row
        for row in selected
    }
    expected = 3 * 12 * 3
    assert len(selected) == expected, f"Expected {expected} in-domain test rows; found {len(selected)}"

    lines = [
        "% Generated from PAIRED_TESTS_proposed_vs_others.csv (canonical v2).",
        r"\begin{table}[htbp]",
        r"\centering",
        original_caption("tab:t_test_results"),
        r"\label{tab:t_test_results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"\textbf{Dataset} & \textbf{Competitor} & $t_{\mathrm{MAE}}$ & "
            r"$p_{\mathrm{MAE,Holm}}$ & $t_{\mathrm{RMSE}}$ & $p_{\mathrm{RMSE,Holm}}$ \\"
        ),
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(("EPFL", "NFFA", "Biofilms")):
        if dataset_index:
            lines.append(r"\midrule")
        for model in PAIR_ORDER:
            mae = by_key[(dataset, model, "mae")]
            rmse = by_key[(dataset, model, "rmse")]
            lines.append(
                f"{DATASET_DISPLAY[dataset]} & {MODEL_SHORT[model]} & "
                f"${float(mae['t_stat']):.2f}$ & {p_cell(float(mae['p_paired_t_holm']))} & "
                f"${float(rmse['t_stat']):.2f}$ & {p_cell(float(rmse['p_paired_t_holm']))} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize\textit{Note:} Reported $p$-values are Holm-adjusted within each dataset--metric family, and bold values are significant at $\alpha=0.05$. Differences are calculated as HSCF-KANet minus competitor. A negative $t$ statistic therefore favours HSCF-KANet for MAE or RMSE, whereas a positive statistic favours the competitor.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


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
    lines = [
        "% Generated from HSCFKANET_FAILURE_CASES.csv (canonical v2).",
        "% Case-to-image mapping:",
    ]
    for dataset, image_id, case in selections:
        assert (dataset, image_id) in by_key, f"Missing failure case: {dataset}/{image_id}"
        lines.append(f"% {case}: {image_id}")
    lines.extend(
        [
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            (
                r"\caption{Representative in-domain failure cases of HSCF-KANet. "
                r"Predicted SNR is averaged across the five independently trained seed models; "
                r"signed error is predicted minus target SNR.}"
            ),
            r"\label{tab:hscf_failure_cases}",
            r"\small",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            (
                r"\textbf{Dataset} & \textbf{Case} & \textbf{Target SNR} & "
                r"\textbf{Predicted SNR} & \textbf{Signed error} & \textbf{Absolute error} \\"
            ),
            r"& & \textbf{(dB)} & \textbf{(dB)} & \textbf{(dB)} & \textbf{(dB)} \\",
            r"\midrule",
        ]
    )
    for dataset, image_id, case in selections:
        row = by_key[(dataset, image_id)]
        lines.append(
            f"{DATASET_DISPLAY[dataset]} & {case} & "
            f"{float(row['target_snr_db']):.4f} & {float(row['predicted_snr_db']):.4f} & "
            f"{float(row['signed_error_db']):.4f} & {float(row['absolute_error_db']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_readme() -> str:
    lines = [
        "# Canonical v2 LaTeX replacement tables",
        "",
            "These files were generated from `HSCF_results_v2_20260727`. The manuscript",
            "`v2/SCv1/modified.tex` and its bibliography were **not modified**.",
            "All 11 replacement tables retain their original manuscript captions verbatim.",
            "Ablation variants appear only in the three ablation tables; the in-domain and",
            "cross-dataset comparison tables contain only standard/external baselines plus",
            "HSCF-KANet.",
        "",
        "Replace the complete existing `\\begin{table} ... \\end{table}` block that has",
        "the matching label. File 12 is a new insertion rather than a replacement.",
        "",
        "| File | Label | Action |",
        "|---|---|---|",
    ]
    for filename, label, action in TABLE_MAP:
        lines.append(f"| `{filename}` | `{label}` | {action} |")
    lines.extend(
        [
            "",
            "## Source rules used",
            "",
            "- In-domain means/SDs: `statistics/STATS_indomain_mean_sd_ci.csv`.",
            "- Cross-dataset means/SDs: `statistics/STATS_crossdataset_mean_sd_ci.csv`.",
            "- Paired tests: `statistics/PAIRED_TESTS_proposed_vs_others.csv`; the table uses",
            "  per-dataset paired seed tests and Holm-adjusted p-values.",
            "- Inference timing: only `metadata/INFERENCE_BENCHMARK.csv`.",
            "- Serialized size: `metadata/CHECKPOINT_HASHES.csv`.",
            "- Failure cases: `predictions/HSCFKANET_FAILURE_CASES.csv`.",
            "",
            "## Two manual follow-ups",
            "",
            "1. Append the two entries in `sota_iqa_bibliography_entries.bib` to the manuscript",
            "   bibliography if those citation keys are not already present.",
            "2. After replacing Table `tab:model_complexity`, revise the nearby prose that still",
            "   discusses the old training-time/GFLOP columns. Canonical v2 supports controlled",
            "   inference time and true checkpoint size; it does not provide a unified new",
            "   training-time or GFLOP audit.",
            "",
            "`_preview_all_tables.tex` is a standalone preview document, not manuscript content.",
        ]
    )
    return "\n".join(lines)


def make_preview() -> str:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{array,graphicx,booktabs,multirow,amsmath}",
        r"\providecommand{\citep}[1]{[#1]}",
        r"\begin{document}",
        r"\section*{Canonical v2 replacement-table preview}",
    ]
    for filename, _, _ in TABLE_MAP:
        lines.extend([rf"\input{{{filename}}}", r"\clearpage"])
    lines.append(r"\end{document}")
    return "\n".join(lines)


def validate_inputs(
    indomain: list[dict[str, str]],
    cross: list[dict[str, str]],
    timings: list[dict[str, str]],
    checkpoints: list[dict[str, str]],
) -> None:
    assert len(indomain) == 39, f"Expected 39 in-domain summary rows; found {len(indomain)}"
    assert len(cross) == 78, f"Expected 78 cross-dataset summary rows; found {len(cross)}"
    assert len(timings) == 26, f"Expected 26 benchmark rows; found {len(timings)}"
    assert len(checkpoints) == 195, f"Expected 195 checkpoint rows; found {len(checkpoints)}"
    for row in indomain + cross:
        assert int(row["n_seeds"]) == len(SEEDS), f"Unexpected n_seeds in {row}"
    hashes = {row["sha256"] for row in checkpoints}
    assert len(hashes) == 195, "Checkpoint hashes are not all unique"


def validate_outputs() -> None:
    """Perform structural checks that do not require a local TeX installation."""
    labels: list[str] = []
    for filename, label, _ in TABLE_MAP:
        path = OUT / filename
        text = path.read_text(encoding="utf-8")
        assert text.count(r"\begin{table}") == 1
        assert text.count(r"\end{table}") == 1
        assert text.count(r"\begin{tabular}") == 1
        assert text.count(r"\end{tabular}") == 1
        assert text.count(rf"\label{{{label}}}") == 1
        if label in ORIGINAL_CAPTIONS:
            assert text.count(original_caption(label)) == 1
        labels.append(label)

        row_count = sum(line.rstrip().endswith(r"\\") for line in text.splitlines())
        assert row_count == EXPECTED_LATEX_ROWS[filename], (
            f"Unexpected row count in {filename}: {row_count}"
        )

        # All generated table fragments use braces only for LaTeX grouping;
        # comments are excluded before checking the balance.
        uncommented = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
        depth = 0
        for char in uncommented:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                assert depth >= 0, f"Closing brace underflow in {filename}"
        assert depth == 0, f"Unbalanced braces in {filename}: depth={depth}"

    assert len(labels) == len(set(labels)), "Generated LaTeX labels are not unique"


def main() -> None:
    indomain_rows = read_csv(INDOMAIN_CSV)
    cross_rows = read_csv(CROSS_CSV)
    paired_rows = read_csv(PAIRED_CSV)
    timing_rows = read_csv(TIMING_CSV)
    checkpoint_rows = read_csv(CHECKPOINT_CSV)
    failure_rows = read_csv(FAILURE_CSV)
    validate_inputs(indomain_rows, cross_rows, timing_rows, checkpoint_rows)

    indomain = {
        (row["source_dataset"], row["architecture"]): row for row in indomain_rows
    }
    cross = {
        (row["source_dataset"], row["test_dataset"], row["architecture"]): row
        for row in cross_rows
    }

    write_text("01_in_domain_epfl.tex", make_full_table("EPFL", "tab:all_performance_EPFL", indomain))
    write_text("02_ablation_epfl.tex", make_ablation_table("EPFL", "tab:abltation_EPFL", indomain))
    write_text(
        "03_cross_dataset_train_epfl.tex",
        make_cross_table("EPFL", ("NFFA", "Biofilms"), "tab:cross_epfl_test_nffa_biofilm", cross),
    )
    write_text("04_in_domain_nffa.tex", make_full_table("NFFA", "tab:all_performance_NFFA", indomain))
    write_text("05_ablation_nffa.tex", make_ablation_table("NFFA", "tab:ablation_nffa", indomain))
    write_text(
        "06_cross_dataset_train_nffa.tex",
        make_cross_table("NFFA", ("EPFL", "Biofilms"), "tab:cross_nffa_test_epfl_biofilm", cross),
    )
    write_text(
        "07_in_domain_biofilms.tex",
        make_full_table("Biofilms", "tab:all_performance_biofilm", indomain),
    )
    write_text(
        "08_ablation_biofilms.tex",
        make_ablation_table("Biofilms", "tab:ablation_biofilm", indomain),
    )
    write_text(
        "09_cross_dataset_train_biofilms.tex",
        make_cross_table("Biofilms", ("EPFL", "NFFA"), "tab:cross_biofilm_test_epfl_nffa", cross),
    )
    write_text("10_model_complexity.tex", make_complexity_table(timing_rows, checkpoint_rows))
    write_text("11_paired_tests_in_domain.tex", make_paired_table(paired_rows))
    write_text("12_failure_cases.tex", make_failure_table(failure_rows))
    write_text(
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
    write_text("README.md", make_readme())
    write_text("_preview_all_tables.tex", make_preview())

    generated = [filename for filename, _, _ in TABLE_MAP]
    for filename in generated:
        assert (OUT / filename).is_file()
    validate_outputs()
    print(f"Generated and validated {len(generated)} replacement/insert tables in {OUT}")


if __name__ == "__main__":
    # Compatibility entry point: the manuscript-style generator preserves the
    # exact table blocks from modified.tex and supersedes the earlier redesign.
    from generate_manuscript_style_tables import main as manuscript_style_main

    manuscript_style_main()

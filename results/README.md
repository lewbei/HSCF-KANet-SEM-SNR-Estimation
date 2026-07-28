# HSCF Evaluation Results — Canonical File Set (v2, 2026-07-27)

All results produced AFTER the checkpoint de-duplication fix (195/195 unique
SHA-256, see metadata/CHECKPOINT_HASHES.csv). Older files live in
`extracted_dataset/archive_deprecated_results/` in the project — do NOT use them.

## v2 corrections (reviewer round 2)
1. `STATS_indomain_overall_mean_sd_ci.csv` recomputed consistently: n=15 pooled
   runs use df=14, t(14)=2.145 with sqrt(15). Treated as DESCRIPTIVE only —
   seeds repeat across datasets, so the canonical inferential units remain the
   per-dataset CIs in `STATS_indomain_mean_sd_ci.csv` (n=5, t(4)=2.776, sqrt(5)).
2. `PAIRED_TESTS_proposed_vs_others.csv` regenerated WITHOUT pooling:
   - in-domain: per dataset, 5 paired seed values (3 datasets × 12 competitors × 3 metrics)
   - cross-dataset: per source→target direction, 5 paired seed values (6 directions × 12 × 3)
   - Holm-Bonferroni applied within each declared family (competitors within
     dataset-or-direction × metric). Note: with n=5 pairs, Wilcoxon two-sided
     cannot reach p<0.05 (min exact p=0.0625); paired t-test is primary.
3. Inference times: cite ONLY `INFERENCE_BENCHMARK.csv`. Timing columns inside
   ALL_MODELS_PER_SEED_RESULTS.csv and any older evaluation CSVs are diagnostics
   from mixed protocols — never cite them. `model_comparison.csv` now carries
   benchmark timings (bench_bs1_median_ms, bench_bs32_ms_per_sample) instead.
4. CNN_RF size: `model_comparison.csv` reports true serialized checkpoint sizes
   (checkpoint_mb_mean/min/max, includes the Random Forest: Biofilms 7.36–7.41 MB,
   EPFL 115.09–115.25 MB, NFFA 104.18–104.31 MB). The column cnn_param_memory_mb
   is explicitly CNN parameter memory ONLY (0.089 MB), excluding the RF.
5. HSCF-KANet failure cases: `HSCFKANET_FAILURE_CASES.csv` (top-20 worst in-domain
   images per dataset, mean over 5 seeds + per-seed predictions). Full per-image
   predictions (63,000 rows: 15 checkpoints × 3 test sets) in
   `PER_IMAGE_PREDICTIONS_proposed_HSCFKANet.csv`. Note: the earlier
   "evaluation_output_v2" predictions were not present in the project; these
   predictions were regenerated from the same checkpoints with the same code.

## Primary results
| File | Contents |
|---|---|
| `per_seed/ALL_MODELS_PER_SEED_RESULTS.csv` | 585 rows: 13 architectures × 3 source datasets × 5 seeds × 3 test sets. One row per seed |
| `per_seed/PER_SEED_*.csv` | Wide format, seed42..seed1024 as columns |
| `statistics/STATS_indomain_mean_sd_ci.csv` | CANONICAL: mean ± sample SD + 95% CI per architecture × dataset (n=5, t(4)=2.776) |
| `statistics/STATS_crossdataset_mean_sd_ci.csv` | same, per architecture × source × test |
| `statistics/STATS_indomain_overall_mean_sd_ci.csv` | DESCRIPTIVE pooled summary (n=15, t(14)=2.145) |
| `statistics/PAIRED_TESTS_proposed_vs_others.csv` | 324 rows: per-dataset and per-direction paired t + Wilcoxon, Holm-corrected within family |
| `metadata/CHECKPOINT_HASHES.csv` | 195 checkpoints: sha256 (all unique) + true file size MB |
| `metadata/INFERENCE_BENCHMARK.csv` + `_protocol.json` | ONLY citable timing source — identical protocol, all 13 architectures |
| `predictions/PER_IMAGE_PREDICTIONS_proposed_HSCFKANet.csv` | 63,000 per-image predictions (HSCF-KANet) |
| `predictions/HSCFKANET_FAILURE_CASES.csv` | 60 rows: worst 20 in-domain failures per dataset |
| `evaluation_raw/` | full per-evaluation CSVs, summaries, 495+ plots |
| `sota_baselines/` | ARNIQA/TOPIQ run metrics + per-image predictions |
| `scripts/` | all evaluation/benchmark/prediction scripts |

## Inference timing protocol (INFERENCE_BENCHMARK.csv)
- GPU: NVIDIA GeForce RTX 4060 8 GB; torch 2.12.0+cu130
- Input: synthetic 1×1×256×256 tensor (dataset images are 256×256 grayscale)
- Batch sizes 1 and 32; 50 warm-up + 200 timed iterations
- torch.cuda.synchronize() per iteration; forward pass only, no image I/O
- CNN_RF includes RandomForestRegressor.predict (full inference path)
- Weights: Biofilms/seed42 per architecture (timing is weight-independent)

## Evaluation protocol
- Test sets: Biofilms 140, EPFL 2130, NFFA 1930 images (256×256, SNR in dB)
- Metrics: MAE, MSE, RMSE, R² (sklearn)
- In-domain = source == test dataset; cross-dataset otherwise
- `proposed` architecture = HSCF-KANet (RawPlusPSDAndNVarScalar, ksizes=(3,5,7,9))

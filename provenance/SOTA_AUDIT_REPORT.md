# Audit Report: ARNIQA-SNR and TOPIQ-NR-SNR Evaluation

**Generated:** 2026-07-22 19:06:05.518640

## Validation Checks

| Check | Result |
|-------|--------|
| run_metrics.csv has 90 rows | ✅ PASS |
| per_image_predictions.csv has 126000 rows | ✅ PASS |
| summary has 54 rows | ✅ PASS |
| All groups have 5 seeds | ✅ PASS |
| 30 distinct checkpoints | ✅ PASS |
| 30 unique SHA256 hashes | ✅ PASS |
| EPFL test count = 2130 | ✅ PASS |
| NFFA test count = 1930 | ✅ PASS |
| Biofilms test count = 140 | ✅ PASS |
| All metrics match within 1e-5 | ✅ PASS |
| No NaN or infinite values | ✅ PASS |

## Summary Statistics

- Total run_metrics rows: 90
- Total prediction rows: 126000
- Distinct checkpoints: 30
- Unique SHA256 hashes: 30
- All metric recomputations match: True

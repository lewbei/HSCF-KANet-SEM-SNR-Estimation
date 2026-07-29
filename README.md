# HSCF-KANet for Blind SNR Estimation in SEM Images

Official code and canonical five-seed results for the manuscript **“Blind SNR Estimation in SEM Images Using Hybrid Statistical and CNN Features.”**

HSCF-KANet combines a lightweight CNN backbone with multi-scale local-variance and power-spectral-density (PSD) descriptors, followed by a Kolmogorov–Arnold Network (KAN) regression head. The repository also contains the ablation models, conventional deep-learning baselines, ARNIQA-SNR and TOPIQ-NR-SNR wrappers, evaluation scripts, and the corrected 2026-07-27 result package.

## Release status

- Canonical result version: `v2`, generated on 2026-07-27.
- Random seeds: `42`, `123`, `456`, `789`, and retrained `1024`.
- Evaluation coverage: 13 architectures, 3 source datasets, 3 test datasets, and 5 seeds (585 per-seed rows).
- Checkpoint audit: 195/195 SHA-256 hashes are unique in `results/metadata/CHECKPOINT_HASHES.csv`.
- Statistical testing: seed-matched, two-tailed paired Student's t-tests with Holm–Bonferroni correction within each dataset/direction–metric family.
- Inference timing: use only `results/metadata/INFERENCE_BENCHMARK.csv`; it follows one controlled protocol for all architectures.

## Repository contents

| Path | Contents |
|---|---|
| `model/` | HSCF-KANet, KAN/MLP/RF ablations, and conventional regression architectures |
| `model/sota/` | ARNIQA-SNR and TOPIQ-NR-SNR regression wrappers |
| `training/` | Five-seed training entry points |
| `evaluation/` | In-domain, cross-dataset, per-image, and inference benchmark scripts |
| `data/` | Gaussian-noise dataset preparation and verification utilities |
| `analysis/` | Classical estimators, statistical recomputation, complexity, and LaTeX-table utilities |
| `results/` | Canonical per-seed metrics, confidence intervals, paired tests, timing, hashes, and HSCF-KANet predictions |
| `provenance/` | SOTA training provenance and audit reports |
| `third_party/` | Pinned upstream dependency instructions |

Raw SEM images, trained checkpoints, generated plots, caches, and bundled copies of external repositories are intentionally excluded. Dataset redistribution is subject to each dataset provider's terms. Checkpoints and the complete archival result bundle should be deposited with the associated DOI release.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The SOTA wrappers use pinned upstream repositories rather than vendored copies. Install them as described in [`third_party/README.md`](third_party/README.md). No pretrained weights were used for ARNIQA-SNR or TOPIQ-NR-SNR.

## Trained checkpoints

The five-seed checkpoints are stored in three private Hugging Face model repositories:

- [EPFL checkpoints](https://huggingface.co/lew96123/HSCF-KANet-EPFL)
- [NFFA-EUROPE checkpoints](https://huggingface.co/lew96123/HSCF-KANet-NFFA)
- [Biofilm checkpoints](https://huggingface.co/lew96123/HSCF-KANet-Biofilms)

Authenticate with a Hugging Face account that has access, then download each repository into the dataset directory expected by the evaluation scripts:

```bash
hf auth login
hf download lew96123/HSCF-KANet-EPFL --local-dir downloaded_models/EPFL
hf download lew96123/HSCF-KANet-NFFA --local-dir downloaded_models/NFFA
hf download lew96123/HSCF-KANet-Biofilms --local-dir downloaded_models/Biofilms
```

Each repository contains one top-level folder per architecture. This includes `ARNIQA_SNR/` and `TOPIQ_NR_SNR/`; neither SOTA architecture is nested under an additional `checkpoints/` directory.

## Dataset layout

The PyTorch models expect the following split layout:

```text
Dataset/
├── Biofilms/
│   ├── train/labels.csv
│   ├── val/labels.csv
│   └── test/labels.csv
├── EPFL/
│   ├── train/labels.csv
│   ├── val/labels.csv
│   └── test/labels.csv
└── NFFA/
    ├── train/labels.csv
    ├── val/labels.csv
    └── test/labels.csv
```

Each split CSV uses `filename` and `snr_db`. Images may be stored as `.pt`, `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`. The SOTA training entry point additionally accepts one dataset-level `labels.csv` containing `split`, `dataset`, `noisy_image`, and `SNR_classical_dB`.

## Training

Run the main five-seed trainer from the repository root. Select the dataset with `HSCF_DATASET_DIR`:

```bash
HSCF_DATASET_DIR=Dataset/Biofilms python3 training/train_all_models.py
```

The default seeds and optimisation settings match the manuscript. W&B logging is disabled by default and can be enabled with `HSCF_USE_WANDB=1`.

The generic PyTorch trainer excludes CNN-RF because that baseline uses a two-stage CNN-feature/Random-Forest fitting procedure. Its architecture, serialized-checkpoint evaluation path, per-seed metrics, checkpoint hashes, and full inference timing are included. The exact standalone CNN-RF fitting entry point was not present in the retained final workspace and is therefore not represented as original experiment code.

Train ARNIQA-SNR and TOPIQ-NR-SNR from random initialisation:

```bash
python3 training/train_sota_iqa.py \
  --dataset-root Dataset \
  --output-root outputs/sota_iqa \
  --model all \
  --dataset all
```

The wrappers explicitly set all pretrained-weight options to false or `None`; see `provenance/SOTA_TRAINING_PROVENANCE.md`.

## Evaluation and statistics

Evaluation scripts expect model checkpoints below `downloaded_models/<source>/<architecture>/` and dataset splits below `Dataset/`:

```bash
python3 evaluation/evaluate_all_models.py
python3 evaluation/evaluate_missing_models.py
python3 evaluation/predict_hscf_kanet.py
```

Recompute the mean, sample standard deviation, 95% confidence intervals, paired tests, and Holm-adjusted p-values directly from the 585 canonical per-seed rows:

```bash
python3 analysis/recompute_statistics.py
```

By default, recomputed files are written to `recomputed_statistics/`, leaving the canonical files unchanged.

## Canonical results

The main machine-readable source is `results/per_seed/ALL_MODELS_PER_SEED_RESULTS.csv`. Important derived files are:

- `results/statistics/STATS_indomain_mean_sd_ci.csv`
- `results/statistics/STATS_crossdataset_mean_sd_ci.csv`
- `results/statistics/PAIRED_TESTS_proposed_vs_others.csv`
- `results/metadata/INFERENCE_BENCHMARK.csv`
- `results/predictions/HSCFKANET_FAILURE_CASES.csv`

See [`results/README.md`](results/README.md) for the complete protocol and file definitions.

## Reproducibility notes

- In-domain confidence intervals use five independent seed runs (`n=5`, `df=4`).
- The pooled 15-run in-domain summary is descriptive only because the same seeds recur across datasets.
- Paired tests are never pooled across datasets or transfer directions.
- CNN-RF timing includes both CNN feature extraction and `RandomForestRegressor.predict`.
- Checkpoint-size values for CNN-RF include the serialized Random Forest and vary by dataset and seed.
- The project focuses on controlled additive Gaussian noise; real SEM shot noise, charging artefacts, and mixed noise remain outside the current experimental scope.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The reserved archival code DOI is [10.5281/zenodo.21661984](https://doi.org/10.5281/zenodo.21661984). The final journal DOI will be added after publication.

## License

The software is released under the MIT License. See [`LICENSE`](LICENSE).

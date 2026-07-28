# Source manifest

This repository was assembled from the final local manuscript workspace on 2026-07-28.

## Included source groups

- HSCF-KANet and standard comparison models: `code/model/`
- Ablation definitions: `code/alba/`
- Main five-seed trainer: `code/train_code.py`
- Data utilities: `code/utils/`
- Canonical evaluation scripts: `HSCF_results_v2_20260727/scripts/`
- SOTA wrappers and provenance: `sota_iqa_final_v2/`
- Manuscript-table utilities: workspace `scripts/`
- Canonical result package: `HSCF_results_v2_20260727/`

Files were renamed only where necessary to use valid Python module names, namely `CNN + MLP` to `CNN_MLP.py` and `CNN-RF` to `CNN_RF.py`. Portable path defaults and command-line output paths were added in the public copy. The model computations and published result CSVs were not changed.

## Excluded content

- Raw datasets and generated noisy images
- Model checkpoints and experiment-tracker data
- Deprecated result folders and superseded archives
- Hundreds of generated diagnostic plots
- Local environment files and credentials
- Bundled Git histories and source copies of ARNIQA and IQA-PyTorch

The full archival result bundle and checkpoint collection should be published separately through the DOI-bearing record associated with the manuscript.

## Retained-source limitations

The final workspace retained the CNN-RF architecture and its complete checkpoint evaluation path, but not the exact standalone two-stage fitting script. It also did not contain the CalibNet architecture definition. These omissions are stated in the public documentation rather than replacing them with reconstructed code that might differ from the reported experiments.

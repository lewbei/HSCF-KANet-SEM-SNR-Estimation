# Data preparation

Raw SEM images are not redistributed in this repository. Obtain each source dataset from its provider and follow the applicable terms of use.

`prepare_dataset_with_noise.py` implements deterministic clean-image splitting followed by the manuscript's controlled Gaussian corruption protocol. It uses noise variances from 0.001 to 0.010 by default and computes the operational global amplitude-ratio SNR from each corrupted image. All noise variants derived from one clean reference image remain in the same split to prevent reference-image leakage.

For example:

```bash
python3 data/prepare_dataset_with_noise.py \
  --images path/to/clean/biofilm/images \
  --output-root Dataset/Biofilms \
  --dataset-name Biofilms

python3 data/verify_noise.py Dataset/Biofilms
```

`verify_noise.py` checks saved tensor shapes and ranges, recomputes every SNR label, confirms that every reference has all ten variance levels, and detects reference-image leakage across splits.

Expected split CSV columns:

```text
filename,snr_db,split
relative/path/to/image.pt,10.0,train
```

The `split` column may be omitted when each split has its own `labels.csv`.

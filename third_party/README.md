# Third-party model sources

The upstream repositories are not vendored here. Clone the exact audited commits from the repository root:

```bash
mkdir -p model/sota/github
git clone https://github.com/miccunifi/ARNIQA model/sota/github/ARNIQA
git -C model/sota/github/ARNIQA checkout 66d16eb0ff1e1655872d32c0c233614a3922aaada

git clone https://github.com/chaofengc/IQA-PyTorch model/sota/github/IQA-PyTorch
git -C model/sota/github/IQA-PyTorch checkout 18dd7a19694e94aac21019170e3f5e63d6b4e19e
```

Install any additional dependencies declared by those projects. Their original licenses apply to their source code.

The local wrappers are `model/sota/ARNIQAWrapper.py` and `model/sota/TOPIQNRWrapper.py`. Both were configured for training from random initialisation. No upstream pretrained weights were downloaded or loaded.

CalibNet results are included in the canonical result tables. Re-evaluating its original checkpoints requires the matching `CalibNet.py` architecture definition to be placed in `model/`; that source was not part of the final local code snapshot used to assemble this public release.

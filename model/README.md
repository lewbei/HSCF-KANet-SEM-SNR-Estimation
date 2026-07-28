# Model definitions

| File | Architecture in the result CSVs |
|---|---|
| `proposed.py` | `proposed` (HSCF-KANet) |
| `CNN_KAN.py` | `CNN_KAN` |
| `CNN_KAN_Local.py` | `CNN_KAN_Local` |
| `CNN_KAN_PSD.py` | `CNN_KAN_PSD` |
| `CNN_MLP.py` | `CNN_MLP` |
| `CNN_RF.py` | `CNN_RF` feature extractor |
| `ResNet18Regression.py` | `ResNet18Regression` |
| `ResNeXt50Regression.py` | `ResNeXt50Regression` |
| `ConvNextTinyRegression.py` | `ConvNextTinyRegression` |
| `ViTRegression.py` | `ViTRegression` |
| `sota/ARNIQAWrapper.py` | `ARNIQA_SNR` |
| `sota/TOPIQNRWrapper.py` | `TOPIQ_NR_SNR` |

The dynamic training/evaluation scripts inject their selected `device` into model modules before loading them. This preserves compatibility with the source files and the checkpoint naming used by the experiments.

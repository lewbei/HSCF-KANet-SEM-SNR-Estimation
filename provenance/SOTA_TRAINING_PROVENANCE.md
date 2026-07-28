# Training Provenance Report

**Generated:** 2026-07-22 19:06:35.307590

## File Hashes

| File | Path | SHA256 |
|------|------|--------|
| ARNIQAWrapper.py | /mnt/d/hscf/code/model/ARNIQAWrapper.py | 54128fe068d0062d4c008d11201173119ac0c281f871487d76baea82ba9707fe |
| TOPIQNRWrapper.py | /mnt/d/hscf/code/model/TOPIQNRWrapper.py | 6ddeae09777ce15423a71b77b0e9cbf70f26dcea935a6e6a90db99ba4194252c |
| train_and_eval.py | /mnt/d/hscf/sota_iqa_baselines/src/train_and_eval.py | abb017dee326c262233095b2b36cf7a5e9f6feda8ade88ee8404dd8443bf0164 |

## Upstream Repositories

| Repository | URL | Commit |
|------------|-----|--------|
| ARNIQA | https://github.com/miccunifi/ARNIQA | 66d16eb0ff1e1655872d32c0c233614a3922aaad |
| IQA-PyTorch (TOPIQ) | https://github.com/chaofengc/IQA-PyTorch | 18dd7a19694e94aac21019170e3f5e63d6b4e19e |

## Pretrained Weight Settings

| Model | Setting | Value |
|-------|---------|-------|
| ARNIQA-SNR | pretrained | False |
| ARNIQA-SNR | weights | None (set by upstream when pretrained=False) |
| TOPIQ-NR-SNR | backbone_pretrain | False |
| TOPIQ-NR-SNR | pretrained | False |
| TOPIQ-NR-SNR | pretrained_model_path | None |

## Training Configuration

- **Preprocessing:** Grayscale images scaled to [0, 1], converted to 3 channels
- **Gaussian noise range:** 0.001 to 0.010 (variance)
- **Input size:** 256 × 256 pixels
- **Batch size:** 16
- **Optimizer:** AdamW
- **Learning rate:** 0.001
- **Weight decay:** 1e-5
- **Maximum epochs:** 100
- **Loss function:** L1 Loss
- **Early stopping:** Best validation RMSE checkpoint
- **Model selection:** Lowest validation RMSE (test data never used for selection)
- **Dataset splits:** Pre-defined in labels.csv, not randomly created

## Audit Checks

| Check | Status |
|-------|--------|
| ARNIQA pretrained=False | ✅ PASS |
| ARNIQA no weights param | ✅ PASS |
| ARNIQA no torch.hub | ✅ PASS |
| ARNIQA no state_dict_from_url | ✅ PASS |
| TOPIQ backbone_pretrain=False | ✅ PASS |
| TOPIQ pretrained=False | ✅ PASS |
| TOPIQ pretrained_model_path=None | ✅ PASS |
| TOPIQ no pyiqa.create_metric | ✅ PASS |
| No JitteredSNR | ✅ PASS |
| Separate val/test | ✅ PASS |
| Upstream ARNIQA weights=None | ✅ PASS |

## Issues Found

- ⚠️ Found 'pretrained_model' in TOPIQ wrapper

## Conclusion

**No pretrained weights were used.** Both models were trained from random initialization on the SEM SNR regression task.

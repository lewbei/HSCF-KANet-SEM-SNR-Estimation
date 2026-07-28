"""
Fix-up evaluation for models the generic loader couldn't handle:
  - CNN_KAN_Local  (class RawPlusNVarScalar)
  - CNN_KAN_PSD    (class RawPlusPSDScalar)
  - proposed       (class RawPlusPSDAndNVarScalar)
  - CNN_RF         (CNN_FeatureExtractor + sklearn RandomForestRegressor)
  - CalibNet/NFFA  (checkpoint wraps state_dict under 'model_state_dict')
Produces the same per-seed rows as evaluate_all_models.py and appends them.
"""
import os, sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

DATASETS = ['Biofilms', 'EPFL', 'NFFA']
SEEDS = [42, 123, 456, 789, 1024]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE_DIR = str(REPOSITORY_ROOT / 'downloaded_models')
DATASET_BASE_DIR = str(REPOSITORY_ROOT / 'Dataset')
MODEL_DEFS_DIR = str(REPOSITORY_ROOT / 'model')
RESULTS_DIR = str(REPOSITORY_ROOT / 'evaluation_results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Import SEMDataset from the main script
spec = importlib.util.spec_from_file_location(
    'eval_main', Path(__file__).resolve().parent / 'evaluate_all_models.py'
)
eval_main = importlib.util.module_from_spec(spec)
# Prevent running main() on import: it's guarded by __name__ == '__main__', safe.
spec.loader.exec_module(eval_main)
SEMDataset = eval_main.SEMDataset


def load_module(arch):
    spec = importlib.util.spec_from_file_location(f'model_{arch}', os.path.join(MODEL_DEFS_DIR, f'{arch}.py'))
    mod = importlib.util.module_from_spec(spec)
    mod.device = device
    spec.loader.exec_module(mod)
    return mod


MAIN_CLASS = {
    'CNN_KAN_Local': 'RawPlusNVarScalar',
    'CNN_KAN_PSD': 'RawPlusPSDScalar',
    'proposed': 'RawPlusPSDAndNVarScalar',
    'CNN_RF': 'CNN_FeatureExtractor',
    'CalibNet': 'CalibNet',
}

JOBS = (
    [('CNN_KAN_Local', ds, s) for ds in DATASETS for s in SEEDS] +
    [('proposed', ds, s) for ds in DATASETS for s in SEEDS]
)

# Training code (HF repo) instantiated these with ksizes=(3,5,7,9)
BUILD_KWARGS = {
    'CNN_KAN_Local': dict(ksizes=(3, 5, 7, 9)),
    'proposed': dict(ksizes=(3, 5, 7, 9)),
}


def build_model(arch):
    mod = load_module(arch)
    cls = getattr(mod, MAIN_CLASS[arch])
    return cls(**BUILD_KWARGS.get(arch, {})).to(device)


def load_weights(arch, path, model):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and 'model_state_dict' in obj and not all(
            isinstance(v, torch.Tensor) for v in obj.values()):
        obj = obj['model_state_dict']
    if arch == 'CNN_RF':
        model.load_state_dict(obj['cnn_state_dict'])
        rf = obj['rf_model']
        return model, rf
    model.load_state_dict(obj)
    return model, None


def evaluate_nn(model, loader):
    model.eval()
    yt, yp = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)
            out = model(imgs)
            yp.extend(out.detach().cpu().numpy().flatten())
            yt.extend(labels.numpy().flatten())
    return np.array(yt), np.array(yp), time.time() - t0


def evaluate_rf(model, rf, loader):
    model.eval()
    yt, yp = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)
            feats = model(imgs).cpu().numpy()
            pred = rf.predict(feats)
            yp.extend(pred.flatten())
            yt.extend(labels.numpy().flatten())
    return np.array(yt), np.array(yp), time.time() - t0


def inference_time(fn, model, rf=None):
    dummy = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(20):
            f = model(dummy)
            if rf is not None:
                rf.predict(f.cpu().numpy())
        times = []
        for _ in range(100):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            f = model(dummy)
            if rf is not None:
                rf.predict(f.cpu().numpy())
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    return float(np.median(times)), float(np.std(times))


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loaders = {}
    for ds in DATASETS:
        test_dir = os.path.join(DATASET_BASE_DIR, ds, 'test')
        d = SEMDataset(test_dir, os.path.join(test_dir, 'labels.csv'))
        loaders[ds] = DataLoader(d, batch_size=32, shuffle=False, num_workers=0)
        print(f'{ds} test: {len(d)} samples')

    rows = []
    for i, (arch, src, seed) in enumerate(JOBS, 1):
        wpath = os.path.join(MODEL_BASE_DIR, src, arch, f'{arch}_seed{seed}.pth')
        if not os.path.exists(wpath):
            print(f'[{i}/{len(JOBS)}] MISSING {wpath}')
            continue
        try:
            model = build_model(arch)
            model, rf = load_weights(arch, wpath, model)
        except Exception as e:
            print(f'[{i}/{len(JOBS)}] LOAD FAIL {arch}/{src}/seed{seed}: {e}')
            continue

        params = sum(p.numel() for p in model.parameters())
        med_ms, std_ms = inference_time(None, model, rf)

        for test_ds, loader in loaders.items():
            try:
                if arch == 'CNN_RF':
                    yt, yp, tt = evaluate_rf(model, rf, loader)
                else:
                    yt, yp, tt = evaluate_nn(model, loader)
                mae = mean_absolute_error(yt, yp)
                mse = mean_squared_error(yt, yp)
                rows.append({
                    'timestamp': timestamp,
                    'source_dataset': src,
                    'test_dataset': test_ds,
                    'eval_type': 'in-domain' if src == test_ds else 'cross-dataset',
                    'architecture': arch,
                    'seed': seed,
                    'model_name': f'{arch}_seed{seed}',
                    'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse),
                    'r2': r2_score(yt, yp),
                    'n_samples': len(yt),
                    'total_inference_sec': tt,
                    'avg_inference_ms_sample': tt / len(yt) * 1000,
                    'inference_ms_per_sample': med_ms,
                    'inference_std_ms': std_ms,
                    'total_params': params,
                    'model_size_mb': params * 4 / 1024 / 1024,
                })
                print(f'[{i}/{len(JOBS)}] {arch} {src} seed{seed} -> {test_ds}: '
                      f'MAE={mae:.4f} RMSE={np.sqrt(mse):.4f} R2={r2_score(yt, yp):.4f}', flush=True)
            except Exception as e:
                print(f'[{i}/{len(JOBS)}] EVAL FAIL {arch}/{src}/seed{seed}/{test_ds}: {e}')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    new_df = pd.DataFrame(rows)
    fix_csv = os.path.join(RESULTS_DIR, f'fixup_results_{timestamp}.csv')
    new_df.to_csv(fix_csv, index=False)
    print(f'\nFix-up rows: {len(new_df)} -> {fix_csv}')

    # Merge into the main results file (replace any prior rows for these jobs)
    main_csv = os.path.join(RESULTS_DIR, 'all_evaluation_results.csv')
    main_df = pd.read_csv(main_csv)
    keys = ['architecture', 'source_dataset', 'seed']
    fix_jobs = new_df[keys].drop_duplicates()
    job_set = set(map(tuple, fix_jobs.values.tolist()))
    main_df = main_df[~main_df[keys].apply(lambda r: tuple(r) in job_set, axis=1)]
    merged = pd.concat([main_df, new_df], ignore_index=True)
    merged.to_csv(main_csv, index=False)
    print(f'Updated {main_csv}: {len(merged)} total rows')


if __name__ == '__main__':
    main()

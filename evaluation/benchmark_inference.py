"""
Controlled inference-time benchmark — IDENTICAL protocol for all 13 architectures.

Protocol
--------
- Hardware: single GPU (see output), CUDA, torch as installed
- Input: synthetic random tensor, 1 x 1 x 256 x 256 (matches dataset image size)
- Batch sizes: 1 and 32
- Warm-up: 50 iterations (untimed)
- Timed: 200 iterations
- torch.cuda.synchronize() before start and after each iteration
- Measures forward pass ONLY (no image loading / decoding)
- CNN_RF: CNN feature extraction + RandomForestRegressor.predict (full inference path)
- Weights: Biofilms/seed42 checkpoint per architecture (timing is weight-independent)
- Reports: median, mean, std, p5, p95 per iteration (whole batch), + per-sample median
"""
import os, sys, time, importlib.util, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device('cuda')
REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WARMUP, RUNS = 50, 200
BATCHES = [1, 32]
SEED_CKPT = dict(ds='Biofilms', seed=42)

ARCHS = ['CNN_MLP', 'CNN_KAN', 'CNN_KAN_Local', 'CNN_KAN_PSD', 'CNN_RF',
         'CalibNet', 'ConvNextTinyRegression', 'ResNet18Regression',
         'ResNeXt50Regression', 'ViTRegression', 'proposed',
         'ARNIQA_SNR', 'TOPIQ_NR_SNR']

MAIN_CLASS = {'CNN_KAN_Local': 'RawPlusNVarScalar', 'CNN_KAN_PSD': 'RawPlusPSDScalar',
              'proposed': 'RawPlusPSDAndNVarScalar', 'CNN_RF': 'CNN_FeatureExtractor',
              'CalibNet': 'CalibNet', 'CNN_MLP': None, 'CNN_KAN': None,
              'ConvNextTinyRegression': None, 'ResNet18Regression': None,
              'ResNeXt50Regression': None, 'ViTRegression': None}
KWARGS = {'CNN_KAN_Local': dict(ksizes=(3, 5, 7, 9)), 'proposed': dict(ksizes=(3, 5, 7, 9))}


def load_11(arch):
    """Load one of the 11 proposed-pipeline models."""
    path = os.path.join(REPOSITORY_ROOT, 'downloaded_models', SEED_CKPT["ds"], arch,
                        f'{arch}_seed{SEED_CKPT["seed"]}.pth')
    model_file = os.path.join(REPOSITORY_ROOT, 'model', f'{arch}.py')
    spec = importlib.util.spec_from_file_location(f'm_{arch}', model_file)
    mod = importlib.util.module_from_spec(spec); mod.device = device
    spec.loader.exec_module(mod)
    rf = None
    if arch in ('CNN_MLP', 'CNN_KAN', 'ConvNextTinyRegression', 'ResNet18Regression',
                'ResNeXt50Regression', 'ViTRegression'):
        # first instantiable nn.Module subclass (same heuristic as training scripts)
        model = None
        for name in dir(mod):
            attr = getattr(mod, name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                try:
                    model = attr(); break
                except Exception:
                    continue
    else:
        model = getattr(mod, MAIN_CLASS[arch])(**KWARGS.get(arch, {}))
    obj = torch.load(path, map_location=device, weights_only=False)
    if arch == 'CNN_RF':
        model.load_state_dict(obj['cnn_state_dict']); rf = obj['rf_model']
    elif isinstance(obj, dict) and 'model_state_dict' in obj and not all(isinstance(v, torch.Tensor) for v in obj.values()):
        model.load_state_dict(obj['model_state_dict'])
    else:
        model.load_state_dict(obj)
    return model.to(device).eval(), rf


def load_sota(arch):
    sys.path.insert(0, os.path.join(REPOSITORY_ROOT, 'model', 'sota'))
    path = os.path.join(REPOSITORY_ROOT, 'sota_iqa_baselines', 'checkpoints', arch,
                        SEED_CKPT["ds"], f'seed_{SEED_CKPT["seed"]}', 'best.pt')
    if arch == 'ARNIQA_SNR':
        from ARNIQAWrapper import ARNIQASNRWrapper
        model = ARNIQASNRWrapper()
    else:
        from TOPIQNRWrapper import TOPIQNRSNRWrapper
        model = TOPIQNRSNRWrapper()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), None


def bench(model, rf, batch):
    x = torch.randn(batch, 1, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(WARMUP):
            out = model(x)
            if rf is not None:
                rf.predict(out.cpu().numpy())
        torch.cuda.synchronize()
        times = []
        for _ in range(RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if rf is not None:
                rf.predict(out.cpu().numpy())
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def main():
    print('GPU:', torch.cuda.get_device_name(0))
    rows = []
    for arch in ARCHS:
        try:
            if arch in ('ARNIQA_SNR', 'TOPIQ_NR_SNR'):
                model, rf = load_sota(arch)
            else:
                model, rf = load_11(arch)
        except Exception as e:
            print(f'{arch}: LOAD FAIL {e}'); continue
        params = sum(p.numel() for p in model.parameters())
        for batch in BATCHES:
            try:
                t = bench(model, rf, batch)
                rows.append({'architecture': arch, 'batch_size': batch,
                             'median_ms': np.median(t), 'mean_ms': t.mean(), 'std_ms': t.std(ddof=1),
                             'p5_ms': np.percentile(t, 5), 'p95_ms': np.percentile(t, 95),
                             'median_ms_per_sample': np.median(t) / batch,
                             'total_params': params,
                             'includes_rf_predict': rf is not None})
                print(f'{arch:24s} bs={batch:2d}: median {np.median(t):8.2f} ms '
                      f'({np.median(t)/batch:.3f} ms/sample)  [p5 {np.percentile(t,5):.2f}, p95 {np.percentile(t,95):.2f}]',
                      flush=True)
            except Exception as e:
                print(f'{arch} bs={batch}: BENCH FAIL {e}')
        del model
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv('INFERENCE_BENCHMARK.csv', index=False)
    meta = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'warmup': WARMUP, 'runs': RUNS,
            'input': '1x1x256x256 random tensor', 'sync': 'torch.cuda.synchronize per iteration',
            'checkpoint': f'{SEED_CKPT["ds"]}/seed{SEED_CKPT["seed"]}',
            'note': 'forward pass only, no image loading; CNN_RF includes rf.predict'}
    json.dump(meta, open('INFERENCE_BENCHMARK_protocol.json', 'w'), indent=2)
    print('\nsaved INFERENCE_BENCHMARK.csv + protocol json')


if __name__ == '__main__':
    main()

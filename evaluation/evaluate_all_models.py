"""
Comprehensive Model Evaluation Script
- In-domain test evaluation (all 11 architectures × 5 seeds × 3 datasets)
- Cross-dataset evaluation
- Inference time measurement
- Model size comparison
- Generates plots and CSV results
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASETS = ['Biofilms', 'EPFL', 'NFFA']
ARCHITECTURES = [
    'CNN_MLP', 'ConvNextTinyRegression', 'ResNet18Regression',
    'ResNeXt50Regression', 'ViTRegression',
    'CNN_KAN', 'CNN_KAN_Local', 'CNN_KAN_PSD',
    'CNN_RF', 'CalibNet', 'proposed'
]
SEEDS = [42, 123, 456, 789, 1024]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE_DIR = str(REPOSITORY_ROOT / 'downloaded_models')
DATASET_BASE_DIR = str(REPOSITORY_ROOT / 'Dataset')
MODEL_DEFS_DIR = str(REPOSITORY_ROOT / 'model')
RESULTS_DIR = str(REPOSITORY_ROOT / 'evaluation_results')
PLOTS_DIR = f'{RESULTS_DIR}/plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ============================================================================
# DATASET CLASS
# ============================================================================
class SEMDataset(Dataset):
    def __init__(self, images_dir, labels_csv):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.labels_df)

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pt':
            return torch.load(path, weights_only=True)
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            from PIL import Image
            img = Image.open(path).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        else:
            raise ValueError(f'Unsupported image format: {ext}')

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        label_db = torch.tensor([row['snr_db']], dtype=torch.float32)
        image_path = os.path.join(self.images_dir, filename)
        image_tensor = self._load_image(image_path)
        return image_tensor, label_db


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_architecture(arch_name):
    """Load model class from model definition file."""
    model_file = os.path.join(MODEL_DEFS_DIR, f'{arch_name}.py')
    if not os.path.exists(model_file):
        # Try to find in downloaded_models (some models may not have local defs)
        print(f'  Warning: Model definition not found for {arch_name}')
        return None

    spec = importlib.util.spec_from_file_location(f'model_{arch_name}', model_file)
    module = importlib.util.module_from_spec(spec)
    module.device = device
    spec.loader.exec_module(module)

    # Find model class
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
            try:
                model = attr()
                return model
            except:
                continue
    return None


def load_model_weights(model, weights_path):
    """Load weights into model."""
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# INFERENCE TIME MEASUREMENT
# ============================================================================
def measure_inference_time(model, num_warmup=20, num_runs=100):
    """Measure inference time per sample in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }


# ============================================================================
# MODEL SIZE
# ============================================================================
def get_model_size(model):
    """Calculate model size metrics."""
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    return {
        'total_params': total_params,
        'model_size_mb': model_size_mb
    }


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, test_loader):
    """Evaluate model on test set and return metrics."""
    model.eval()
    y_true, y_pred = [], []

    inference_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            outputs = model(images)
            preds = outputs.detach().cpu().numpy().flatten()
            labels_np = labels.numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(labels_np)
    total_inference_time = time.time() - inference_start

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Per-sample inference time
    n_samples = len(y_true)
    avg_inference_ms = (total_inference_time / n_samples) * 1000

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'n_samples': n_samples,
        'total_inference_sec': total_inference_time,
        'avg_inference_ms': avg_inference_ms,
        'y_true': y_true,
        'y_pred': y_pred
    }


# ============================================================================
# PLOTTING
# ============================================================================
def create_evaluation_plots(y_true, y_pred, model_name, dataset_name, plot_dir):
    """Create and save evaluation plots."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Pred vs True scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, c='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    ax.set_xlabel('True SNR (dB)', fontsize=11)
    ax.set_ylabel('Predicted SNR (dB)', fontsize=11)
    ax.set_title(f'Prediction vs Ground Truth', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Residual vs True
    ax = axes[0, 1]
    ax.scatter(y_true, residuals, alpha=0.4, s=15, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('True SNR (dB)', fontsize=11)
    ax.set_ylabel('Residual (Pred - True)', fontsize=11)
    ax.set_title(f'Residual vs Ground Truth', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residual (dB)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Residual Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Residual boxplot by SNR bins
    ax = axes[1, 1]
    n_bins = 5
    y_pred_bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, y_pred_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    residual_by_bin = [residuals[bin_indices == i] for i in range(n_bins)]
    bin_labels = [f'{y_pred_bins[i]:.1f}-{y_pred_bins[i+1]:.1f}' for i in range(n_bins)]
    ax.boxplot(residual_by_bin, labels=bin_labels, showfliers=True)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted SNR Bin (dB)', fontsize=11)
    ax.set_ylabel('Residual (dB)', fontsize=11)
    ax.set_title(f'Residual by Predicted SNR Bin', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f'{model_name}_{dataset_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('='*80)
    print('COMPREHENSIVE MODEL EVALUATION')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print('='*80)

    # Process each dataset (source = where model was trained)
    for source_dataset in DATASETS:
        print(f'\n{"="*80}')
        print(f'SOURCE DATASET: {source_dataset}')
        print(f'{"="*80}')

        # Load all test datasets for cross-dataset evaluation
        test_loaders = {}
        for test_dataset in DATASETS:
            test_dir = os.path.join(DATASET_BASE_DIR, test_dataset, 'test')
            test_labels = os.path.join(test_dir, 'labels.csv')
            if os.path.exists(test_labels):
                ds = SEMDataset(test_dir, test_labels)
                test_loaders[test_dataset] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
                print(f'  Loaded {test_dataset} test set: {len(ds)} samples')

        # Process each architecture
        for arch_name in ARCHITECTURES:
            print(f'\n  --- {arch_name} ---')

            # Load model architecture
            model_template = load_model_architecture(arch_name)
            if model_template is None:
                print(f'    Skipping {arch_name}: cannot load architecture')
                continue

            # Get model size info
            size_info = get_model_size(model_template)

            # Process each seed
            for seed in SEEDS:
                weights_file = f'{arch_name}_seed{seed}.pth'
                weights_path = os.path.join(MODEL_BASE_DIR, source_dataset, arch_name, weights_file)

                if not os.path.exists(weights_path):
                    print(f'    Skipping {weights_file}: not found')
                    continue

                model_name = f'{arch_name}_seed{seed}'

                # Load weights
                try:
                    model = load_model_architecture(arch_name)
                    model = load_model_weights(model, weights_path)
                except Exception as e:
                    print(f'    Error loading {weights_file}: {e}')
                    continue

                # Measure inference time (once per model)
                inference_info = measure_inference_time(model)

                # Evaluate on all test datasets (in-domain + cross-dataset)
                for test_dataset, test_loader in test_loaders.items():
                    is_indomain = (source_dataset == test_dataset)
                    eval_type = 'in-domain' if is_indomain else 'cross-dataset'

                    try:
                        metrics = evaluate_model(model, test_loader)

                        result = {
                            'timestamp': timestamp,
                            'source_dataset': source_dataset,
                            'test_dataset': test_dataset,
                            'eval_type': eval_type,
                            'architecture': arch_name,
                            'seed': seed,
                            'model_name': model_name,
                            'mae': metrics['mae'],
                            'mse': metrics['mse'],
                            'rmse': metrics['rmse'],
                            'r2': metrics['r2'],
                            'n_samples': metrics['n_samples'],
                            'avg_inference_ms_sample': metrics['avg_inference_ms'],
                            'total_inference_sec': metrics['total_inference_sec'],
                            'inference_ms_per_sample': inference_info['median_ms'],
                            'inference_std_ms': inference_info['std_ms'],
                            'total_params': size_info['total_params'],
                            'model_size_mb': size_info['model_size_mb'],
                        }
                        all_results.append(result)

                        domain_label = '✓' if is_indomain else '→'
                        print(f'    {domain_label} {test_dataset}: MAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}, Inference={inference_info["median_ms"]:.2f}ms')

                        # Save plots for in-domain and cross-dataset
                        if is_indomain or True:  # Save all plots
                            plot_subdir = os.path.join(PLOTS_DIR, source_dataset)
                            os.makedirs(plot_subdir, exist_ok=True)
                            create_evaluation_plots(
                                metrics['y_true'], metrics['y_pred'],
                                model_name, test_dataset, plot_subdir
                            )

                    except Exception as e:
                        print(f'    Error evaluating on {test_dataset}: {e}')

                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    print(f'\n{"="*80}')
    print('SAVING RESULTS')
    print(f'{"="*80}')

    df = pd.DataFrame(all_results)

    # Save raw results
    raw_csv = os.path.join(RESULTS_DIR, f'all_evaluation_results_{timestamp}.csv')
    df.to_csv(raw_csv, index=False)
    print(f'Raw results saved to: {raw_csv}')

    # Also save as latest
    latest_csv = os.path.join(RESULTS_DIR, 'all_evaluation_results.csv')
    df.to_csv(latest_csv, index=False)
    print(f'Latest results saved to: {latest_csv}')

    # ============================================================================
    # SUMMARY TABLES
    # ============================================================================

    # 1. In-domain summary (mean ± std across seeds)
    print(f'\n{"="*80}')
    print('IN-DOMAIN EVALUATION SUMMARY (mean ± std across 5 seeds)')
    print(f'{"="*80}')

    indomain_df = df[df['eval_type'] == 'in-domain'].copy()
    indomain_summary = indomain_df.groupby(['source_dataset', 'architecture']).agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'inference_ms_per_sample': ['mean', 'std'],
        'total_params': 'first',
        'model_size_mb': 'first'
    }).round(4)

    summary_csv = os.path.join(RESULTS_DIR, 'indomain_summary.csv')
    indomain_summary.to_csv(summary_csv)
    print(f'In-domain summary saved to: {summary_csv}')

    # Print summary
    for dataset in DATASETS:
        print(f'\n--- {dataset} ---')
        dataset_df = indomain_df[indomain_df['source_dataset'] == dataset]
        for _, row in dataset_df.groupby('architecture').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'inference_ms_per_sample': 'mean',
            'total_params': 'first'
        }).iterrows():
            pass

    # 2. Cross-dataset summary
    print(f'\n{"="*80}')
    print('CROSS-DATASET EVALUATION SUMMARY')
    print(f'{"="*80}')

    cross_df = df[df['eval_type'] == 'cross-dataset'].copy()
    if len(cross_df) > 0:
        cross_summary = cross_df.groupby(['source_dataset', 'test_dataset', 'architecture']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
        }).round(4)

        cross_csv = os.path.join(RESULTS_DIR, 'cross_dataset_summary.csv')
        cross_summary.to_csv(cross_csv)
        print(f'Cross-dataset summary saved to: {cross_csv}')

    # 3. Model comparison table (averaged across all datasets)
    print(f'\n{"="*80}')
    print('MODEL COMPARISON (averaged across datasets)')
    print(f'{"="*80}')

    model_comparison = indomain_df.groupby('architecture').agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'inference_ms_per_sample': 'mean',
        'total_params': 'first',
        'model_size_mb': 'first'
    }).round(4)

    comparison_csv = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    model_comparison.to_csv(comparison_csv)
    print(f'Model comparison saved to: {comparison_csv}')

    # ============================================================================
    # GENERATE COMPREHENSIVE PLOTS
    # ============================================================================

    # 1. Bar chart: MAE by architecture and dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        dataset_df = indomain_df[indomain_df['source_dataset'] == dataset]
        grouped = dataset_df.groupby('architecture')['mae'].agg(['mean', 'std'])

        colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
        bars = ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'],
                     capsize=5, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MAE (dB)')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('In-Domain MAE by Architecture and Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mae_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Bar chart: R² by architecture and dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        dataset_df = indomain_df[indomain_df['source_dataset'] == dataset]
        grouped = dataset_df.groupby('architecture')['r2'].agg(['mean', 'std'])

        colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
        bars = ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'],
                     capsize=5, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('R²')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('In-Domain R² by Architecture and Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Inference time vs MAE scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for dataset in DATASETS:
        dataset_df = indomain_df[indomain_df['source_dataset'] == dataset]
        grouped = dataset_df.groupby('architecture').agg({
            'inference_ms_per_sample': 'mean',
            'mae': 'mean'
        })
        ax.scatter(grouped['inference_ms_per_sample'], grouped['mae'],
                  label=dataset, s=100, alpha=0.7)
        for arch in grouped.index:
            ax.annotate(arch, (grouped.loc[arch, 'inference_ms_per_sample'],
                              grouped.loc[arch, 'mae']),
                       fontsize=7, alpha=0.7)

    ax.set_xlabel('Inference Time (ms/sample)')
    ax.set_ylabel('MAE (dB)')
    ax.set_title('Inference Time vs MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'inference_vs_mae.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Model size vs MAE scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for dataset in DATASETS:
        dataset_df = indomain_df[indomain_df['source_dataset'] == dataset]
        grouped = dataset_df.groupby('architecture').agg({
            'model_size_mb': 'first',
            'mae': 'mean'
        })
        ax.scatter(grouped['model_size_mb'], grouped['mae'],
                  label=dataset, s=100, alpha=0.7)
        for arch in grouped.index:
            ax.annotate(arch, (grouped.loc[arch, 'model_size_mb'],
                              grouped.loc[arch, 'mae']),
                       fontsize=7, alpha=0.7)

    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('MAE (dB)')
    ax.set_title('Model Size vs MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'size_vs_mae.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Cross-dataset heatmap (if available)
    if len(cross_df) > 0:
        for arch_name in ARCHITECTURES:
            arch_cross = cross_df[cross_df['architecture'] == arch_name]
            if len(arch_cross) == 0:
                continue

            pivot = arch_cross.pivot_table(
                values='mae',
                index='source_dataset',
                columns='test_dataset',
                aggfunc='mean'
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('Test Dataset')
            ax.set_ylabel('Source Dataset')
            ax.set_title(f'Cross-Dataset MAE: {arch_name}')

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                                 ha='center', va='center', color='black', fontsize=10)

            plt.colorbar(im, ax=ax, label='MAE (dB)')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'cross_dataset_{arch_name}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    # ============================================================================
    # FINAL SUMMARY PRINT
    # ============================================================================
    print(f'\n{"="*80}')
    print('EVALUATION COMPLETE')
    print(f'{"="*80}')
    print(f'Total evaluations: {len(all_results)}')
    print(f'Results saved to: {RESULTS_DIR}/')
    print(f'Plots saved to: {PLOTS_DIR}/')
    print(f'\nFiles generated:')
    print(f'  - all_evaluation_results.csv')
    print(f'  - indomain_summary.csv')
    print(f'  - cross_dataset_summary.csv')
    print(f'  - model_comparison.csv')
    print(f'  - plots/ (PNG files)')


if __name__ == '__main__':
    main()

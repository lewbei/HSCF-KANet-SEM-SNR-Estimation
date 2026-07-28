"""
Comprehensive training script for all models and ablation variants.
Includes:
  - All comparison models (ResNet18, ResNeXt50, ConvNeXt Tiny, ViT)
  - All ablation variants (CNN+KAN, CNN+KAN+PSD, CNN+KAN+Local, CNN+MLP, HSCF-KANet)
  - Multi-seed training for confidence intervals (mean ± std)
  - Inference time measurement
  - Model size comparison
  - W&B logging

Usage:
  HSCF_DATASET_DIR=Dataset/Biofilms python3 training/train_all_models.py
"""

import os
import sys
import gc
import time
import importlib.util
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import psutil
import platform
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Dataset path. Override this for EPFL or NFFA.
    "base_dir": os.getenv("HSCF_DATASET_DIR", str(Path("Dataset") / "Biofilms")),

    # Training settings
    "num_epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,

    # Multi-seed settings
    "seeds": [42, 123, 456, 789, 1024],

    # Model directory (all models in single folder)
    "model_dir": os.getenv("HSCF_MODEL_DIR", "model"),
    "checkpoint_dir": os.getenv("HSCF_CHECKPOINT_DIR", "downloaded_models"),

    # Logging
    "use_wandb": os.getenv("HSCF_USE_WANDB", "0") == "1",
    "wandb_project": "HSCF-KANet-SNR-Estimation",
    "wandb_entity": None,  # Set to your wandb username/team if needed
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if os.getenv("HSCF_REQUIRE_CUDA", "0") == "1" and device.type != "cuda":
    raise RuntimeError(
        "HSCF_REQUIRE_CUDA=1 but PyTorch cannot see CUDA. Install a CUDA-enabled torch build."
    )

wandb = None


def setup_wandb():
    """Initialize W&B only when a training run explicitly enables it."""
    global wandb
    if not CONFIG["use_wandb"]:
        return

    wandb_root = os.path.join(os.getcwd(), ".wandb")
    os.makedirs(wandb_root, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", wandb_root)
    os.environ.setdefault("WANDB_CONFIG_DIR", wandb_root)
    os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(wandb_root, "cache"))
    os.environ.setdefault("USERPROFILE", wandb_root)

    try:
        import wandb
        wandb_api_key = os.getenv("WANDB_API")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key, relogin=True)
            print("W&B login completed")
        else:
            print("Warning: WANDB_API not found in .env file")
            CONFIG["use_wandb"] = False
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        CONFIG["use_wandb"] = False
    except Exception as e:
        print(f"Warning: W&B initialization failed: {e}")
        CONFIG["use_wandb"] = False


# ============================================================================
# DATASET CLASSES
# ============================================================================
class SEMDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def _load_image(self, path):
        """Load image from .pt, .png, .jpg, or .jpeg file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pt':
            return torch.load(path, weights_only=True)
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            from PIL import Image
            import numpy as np
            img = Image.open(path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            return torch.from_numpy(img_array).unsqueeze(0)  # Add channel dim [1, H, W]
        else:
            raise ValueError(f"Unsupported image format: {ext}")

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = row['filename'] if 'filename' in row else row['noisy_image']
        snr_value = row['snr_db'] if 'snr_db' in row else row['SNR_classical_dB']
        image_path = os.path.join(self.images_dir, filename)
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(self.images_dir), filename)
        label_db = torch.tensor([snr_value], dtype=torch.float32)
        image_tensor = self._load_image(image_path)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label_db


class JitteredSNRDataset(torch.utils.data.Dataset):
    """On-the-fly SNR down-jitter for zero-mean Gaussian noise."""
    def __init__(self, base_ds, delta_db=4.9, seed=None):
        self.base_ds = base_ds
        self.delta_db = float(delta_db)
        self.seed = seed

    def __len__(self):
        return len(self.base_ds)

    @staticmethod
    def _extra_sigma(img, snr_anchor, snr_target):
        rms = img.float().pow(2).mean().sqrt()
        sigma_anchor = rms / 10**(snr_anchor/20)
        sigma_target = rms / 10**(snr_target/20)
        sigma_add_sq = (sigma_target**2 - sigma_anchor**2).clamp(min=0)
        return sigma_add_sq.sqrt()

    def __getitem__(self, idx):
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)

        img, snr_anchor = self.base_ds[idx]
        if snr_anchor.item() <= 5.1:
            return img, snr_anchor

        eps = torch.rand(1) * self.delta_db
        snr_target = snr_anchor - eps

        img = img.to(torch.float32)
        sigma_add = self._extra_sigma(img, snr_anchor, snr_target)
        sigma_add = sigma_add.to(img.device)
        noise = torch.randn_like(img)
        img_jitter = (img + sigma_add * noise).clamp(0, 1)

        return img_jitter, snr_target


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_from_file(model_file_path):
    """Dynamically load a model from a Python file."""
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    model_module = importlib.util.module_from_spec(spec)
    model_module.device = device

    try:
        spec.loader.exec_module(model_module)

        model = None
        if hasattr(model_module, 'model'):
            model = getattr(model_module, 'model')
        else:
            for attr_name in dir(model_module):
                attr = getattr(model_module, attr_name)
                if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                    try:
                        model = attr().to(device)
                        break
                    except Exception as e:
                        print(f"Failed to instantiate {attr_name}: {e}")
                        continue

        if model is None:
            raise ValueError(f"No model found in {model_file_path}")

        model = model.to(device)
        return model, model.__class__.__name__

    except Exception as e:
        print(f"Error loading model from {model_file_path}: {e}")
        return None, None


# ============================================================================
# INFERENCE TIME MEASUREMENT
# ============================================================================
def measure_inference_time(model, test_loader, device, num_warmup=20, num_runs=100):
    """Measure inference time per sample."""
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
            times.append((time.time() - start) * 1000)  # Convert to ms

    return np.median(times)


# ============================================================================
# MODEL SIZE CALCULATION
# ============================================================================
def get_model_size(model):
    """Calculate model size in MB and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb
    }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_single_model(model, model_name, train_loader, val_loader, test_loader,
                       num_epochs, lr, weight_decay, seed, run_id, wandb_run=None):
    """Train a single model with given seed and return results."""
    total_run_start_time = time.perf_counter()

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = np.inf
    best_model_state = None
    train_loss_curve = []
    val_loss_curve = []
    epoch_times = []

    training_start_time = time.perf_counter()
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()

        # Training
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_curve.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                if images.ndim == 3:
                    images = images.unsqueeze(1)
                labels = labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_curve.append(val_loss)
        epoch_time_seconds = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_time_seconds)

        # Log to W&B per epoch
        if wandb_run and CONFIG["use_wandb"]:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time_seconds": epoch_time_seconds,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    training_time_seconds = time.perf_counter() - training_start_time
    avg_epoch_time_seconds = float(np.mean(epoch_times)) if epoch_times else 0.0

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            outputs = model(images)
            preds = outputs.detach().cpu().numpy().flatten()
            labels = labels.numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Measure inference time
    inference_time_ms = measure_inference_time(model, test_loader, device)
    total_run_time_seconds = time.perf_counter() - total_run_start_time

    # Create loss curve plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss_curve, label="Train Loss")
    ax.plot(val_loss_curve, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE)")
    ax.set_title(f"Training Curves - {model_name} (seed={seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save loss curve plot
    loss_plot_path = f"plots/{model_name}_seed{seed}_loss_curve.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")

    # Log to W&B
    if wandb_run and CONFIG["use_wandb"]:
        wandb_run.log({
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_r2": r2,
            "inference_time_ms": inference_time_ms,
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_seconds / 60,
            "avg_epoch_time_seconds": avg_epoch_time_seconds,
            "total_run_time_seconds": total_run_time_seconds,
            "loss_curve": wandb.Image(loss_plot_path),
        })
    plt.close(fig)

    # Create prediction vs ground truth plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='1:1 Line')
    ax.set_xlabel("True SNR (dB)")
    ax.set_ylabel("Predicted SNR (dB)")
    ax.set_title(f"Prediction vs Ground Truth - {model_name} (seed={seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save pred vs gt plot
    pred_plot_path = f"plots/{model_name}_seed{seed}_pred_vs_gt.png"
    plt.savefig(pred_plot_path, dpi=300, bbox_inches="tight")

    if wandb_run and CONFIG["use_wandb"]:
        wandb_run.log({"pred_vs_gt": wandb.Image(pred_plot_path)})
    plt.close(fig)

    # Create residual vs ground truth plot
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("True SNR (dB)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title(f"Residual vs Ground Truth - {model_name} (seed={seed})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    residual_plot_path = f"plots/{model_name}_seed{seed}_residual_vs_gt.png"
    plt.savefig(residual_plot_path, dpi=300, bbox_inches="tight")

    if wandb_run and CONFIG["use_wandb"]:
        wandb_run.log({"residual_vs_gt": wandb.Image(residual_plot_path)})
    plt.close(fig)

    # Create residual histogram plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Residual (dB)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution - {model_name} (seed={seed})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    hist_plot_path = f"plots/{model_name}_seed{seed}_residual_histogram.png"
    plt.savefig(hist_plot_path, dpi=300, bbox_inches="tight")

    if wandb_run and CONFIG["use_wandb"]:
        wandb_run.log({"residual_histogram": wandb.Image(hist_plot_path)})
    plt.close(fig)

    # Create residual boxplot by binned predictions
    n_bins = 5
    y_pred_bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, y_pred_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    residual_by_bin = [residuals[bin_indices == i] for i in range(n_bins)]
    bin_labels = [f"{y_pred_bins[i]:.1f}-{y_pred_bins[i+1]:.1f}" for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(residual_by_bin, labels=bin_labels, showfliers=True)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Predicted SNR Bin (dB)")
    ax.set_ylabel("Residual (dB)")
    ax.set_title(f"Residual Boxplot by Predicted SNR - {model_name} (seed={seed})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    boxplot_path = f"plots/{model_name}_seed{seed}_residual_boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")

    if wandb_run and CONFIG["use_wandb"]:
        wandb_run.log({"residual_boxplot": wandb.Image(boxplot_path)})
    plt.close(fig)

    return {
        "seed": seed,
        "run_id": run_id,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "best_val_loss": best_val_loss,
        "inference_time_ms": inference_time_ms,
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_seconds / 60,
        "avg_epoch_time_seconds": avg_epoch_time_seconds,
        "total_run_time_seconds": total_run_time_seconds,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    setup_wandb()

    # Setup paths
    base_dir = CONFIG["base_dir"]
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    root_csv = os.path.join(base_dir, 'labels.csv')
    train_csv = os.path.join(train_dir, 'labels.csv')
    val_csv = os.path.join(val_dir, 'labels.csv')
    test_csv = os.path.join(test_dir, 'labels.csv')
    if os.path.exists(root_csv) and not os.path.exists(train_csv):
        train_csv = root_csv
        val_csv = root_csv
        test_csv = root_csv

    # Dataset name for experiment
    dataset_name = Path(base_dir).name
    experiment_name = f"{dataset_name} Multi-Model Comparison"

    print(f"Experiment: {experiment_name}")
    print(f"Seeds: {CONFIG['seeds']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Logging: W&B={CONFIG['use_wandb']}")

    # Load datasets
    print("\nLoading datasets...")
    train_base = SEMDataset(train_dir, train_csv)
    test_set = SEMDataset(test_dir, test_csv)
    val_anchor = SEMDataset(val_dir, val_csv)
    if 'split' in train_base.labels_df.columns:
        train_base.labels_df = train_base.labels_df[train_base.labels_df['split'] == 'train'].reset_index(drop=True)
    if 'split' in val_anchor.labels_df.columns:
        val_anchor.labels_df = val_anchor.labels_df[val_anchor.labels_df['split'] == 'val'].reset_index(drop=True)
    if 'split' in test_set.labels_df.columns:
        test_set.labels_df = test_set.labels_df[test_set.labels_df['split'] == 'test'].reset_index(drop=True)

    # Create data loaders (NO jitter - paper uses no data augmentation)
    train_loader = DataLoader(train_base, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_anchor, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False)

    # Collect all model files from folder (auto-detect)
    model_dir = Path(CONFIG["model_dir"])
    model_files = []

    # Categorize based on filename patterns
    for f in model_dir.glob("*.py"):
        name = f.stem
        if name == "CNN_RF":
            # CNN+RF uses a separate two-stage sklearn training procedure.
            # The retained paper workspace contains its architecture and full
            # evaluation path, but not the exact fitting entry point.
            continue
        if name == "proposed":
            model_files.append(("proposed", f))
        elif name.startswith("CNN_"):
            model_files.append(("ablation", f))
        elif "Regression" in name:
            model_files.append(("comparison", f))
        else:
            model_files.append(("other", f))

    print(f"\nFound {len(model_files)} models:")
    for category, f in model_files:
        print(f"  [{category}] {f.stem}")

    # ============================================================================
    # TRAIN ALL MODELS WITH MULTIPLE SEEDS
    # ============================================================================
    all_results = []
    model_sizes = {}
    run_records = []

    # Create the summary W&B run only after seed runs finish. Per-seed
    # wandb.init(reinit=True) would otherwise finish this run early.
    main_wandb_run = None

    for category, model_file in model_files:
        model_name = model_file.stem
        print(f"\n{'='*80}")
        print(f"Training: {model_name} ({category})")
        print(f"{'='*80}")

        seed_results = []

        for seed in CONFIG["seeds"]:
            print(f"\n  Seed {seed}...")

            # Initialize per-seed W&B run
            seed_wandb_run = None
            if CONFIG["use_wandb"]:
                seed_wandb_run = wandb.init(
                    project=CONFIG["wandb_project"],
                    entity=CONFIG.get("wandb_entity"),
                    name=f"{model_name}_seed{seed}",
                    config={
                        **CONFIG,
                        "model_name": model_name,
                        "category": category,
                        "seed": seed,
                    },
                    group=model_name,
                    job_type="train",
                    reinit=True
                )

            # Load fresh model
            model, _ = load_model_from_file(str(model_file))
            if model is None:
                print(f"  Failed to load model from {model_file}")
                continue

            # Get model size (only once)
            if model_name not in model_sizes:
                size_info = get_model_size(model)
                model_sizes[model_name] = size_info
                # Log model architecture to W&B
                if seed_wandb_run and CONFIG["use_wandb"]:
                    seed_wandb_run.log({
                        "model/total_params": size_info["total_params"],
                        "model/trainable_params": size_info["trainable_params"],
                        "model/model_size_mb": size_info["model_size_mb"],
                    })

            # Train
            result = train_single_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=CONFIG["num_epochs"],
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
                seed=seed,
                run_id=f"{model_name}_seed{seed}",
                wandb_run=seed_wandb_run
            )

            seed_results.append(result)
            checkpoint_dir = (
                Path(CONFIG["checkpoint_dir"])
                / dataset_name
                / model_name
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{model_name}_seed{seed}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    Saved checkpoint: {checkpoint_path}")
            run_records.append({
                "model_name": model_name,
                "category": category,
                "seed": seed,
                "mae": result["mae"],
                "rmse": result["rmse"],
                "r2": result["r2"],
                "best_val_loss": result["best_val_loss"],
                "training_time_seconds": result["training_time_seconds"],
                "training_time_minutes": result["training_time_minutes"],
                "avg_epoch_time_seconds": result["avg_epoch_time_seconds"],
                "total_run_time_seconds": result["total_run_time_seconds"],
                "inference_time_ms": result["inference_time_ms"],
            })
            print(f"    MAE: {result['mae']:.4f} | RMSE: {result['rmse']:.4f} | R²: {result['r2']:.4f}")

            print(f"    Train time: {result['training_time_minutes']:.2f} min | Avg epoch: {result['avg_epoch_time_seconds']:.2f} sec")

            # Save model to W&B
            if seed_wandb_run and CONFIG["use_wandb"]:
                # Save model checkpoint
                model_path = f"models/{model_name}_seed{seed}.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                seed_wandb_run.save(model_path)

                # Log final metrics
                seed_wandb_run.log({
                    "final/test_mae": result["mae"],
                    "final/test_rmse": result["rmse"],
                    "final/test_r2": result["r2"],
                    "final/inference_time_ms": result["inference_time_ms"],
                    "final/training_time_seconds": result["training_time_seconds"],
                    "final/training_time_minutes": result["training_time_minutes"],
                    "final/avg_epoch_time_seconds": result["avg_epoch_time_seconds"],
                    "final/total_run_time_seconds": result["total_run_time_seconds"],
                })
                seed_wandb_run.finish()

            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate results across seeds
        if seed_results:
            mae_values = [r["mae"] for r in seed_results]
            rmse_values = [r["rmse"] for r in seed_results]
            r2_values = [r["r2"] for r in seed_results]
            inference_times = [r["inference_time_ms"] for r in seed_results]
            training_times = [r["training_time_seconds"] for r in seed_results]
            avg_epoch_times = [r["avg_epoch_time_seconds"] for r in seed_results]
            total_run_times = [r["total_run_time_seconds"] for r in seed_results]

            aggregated = {
                "model_name": model_name,
                "category": category,
                "mae_mean": np.mean(mae_values),
                "mae_std": np.std(mae_values),
                "rmse_mean": np.mean(rmse_values),
                "rmse_std": np.std(rmse_values),
                "r2_mean": np.mean(r2_values),
                "r2_std": np.std(r2_values),
                "inference_time_ms": np.median(inference_times),
                "training_time_seconds_mean": np.mean(training_times),
                "training_time_seconds_std": np.std(training_times),
                "training_time_seconds_total": np.sum(training_times),
                "avg_epoch_time_seconds": np.mean(avg_epoch_times),
                "total_run_time_seconds_mean": np.mean(total_run_times),
                "num_seeds": len(seed_results),
                "seed_results": seed_results
            }
            all_results.append(aggregated)

            print(f"\n  Summary for {model_name}:")
            print(f"    MAE:  {aggregated['mae_mean']:.4f} ± {aggregated['mae_std']:.4f}")
            print(f"    RMSE: {aggregated['rmse_mean']:.4f} ± {aggregated['rmse_std']:.4f}")
            print(f"    R²:   {aggregated['r2_mean']:.4f} ± {aggregated['r2_std']:.4f}")
            print(f"    Inference: {aggregated['inference_time_ms']:.2f} ms/sample")
            print(f"    Train time: {aggregated['training_time_seconds_mean'] / 60:.2f} Â± {aggregated['training_time_seconds_std'] / 60:.2f} min/seed")

    # ============================================================================
    # GENERATE SUMMARY TABLE
    # ============================================================================
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        model_name = r["model_name"]
        size_info = model_sizes.get(model_name, {})
        summary_data.append({
            "Model": model_name,
            "Category": r["category"],
            "MAE (dB)": f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}",
            "RMSE (dB)": f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}",
            "R²": f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}",
            "Inference (ms)": f"{r['inference_time_ms']:.2f}",
            "Train Time/Seed (min)": f"{r['training_time_seconds_mean'] / 60:.2f} Â± {r['training_time_seconds_std'] / 60:.2f}",
            "Total Train Time (min)": f"{r['training_time_seconds_total'] / 60:.2f}",
            "Avg Epoch Time (s)": f"{r['avg_epoch_time_seconds']:.2f}",
            "Params (M)": f"{size_info.get('total_params', 0) / 1e6:.2f}",
            "Size (MB)": f"{size_info.get('model_size_mb', 0):.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    if summary_df.empty:
        print("No model results were produced. Check dependency/model loading errors above.")
        return
    summary_df = summary_df.sort_values("Category")

    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv("model_comparison_summary.csv", index=False)
    print("\nSaved: model_comparison_summary.csv")
    run_records_df = pd.DataFrame(run_records)
    run_records_df.to_csv("model_training_runs.csv", index=False)
    print("Saved: model_training_runs.csv")

    # Upload summary to W&B
    if CONFIG["use_wandb"]:
        main_wandb_run = wandb.init(
            project=CONFIG["wandb_project"],
            entity=CONFIG.get("wandb_entity"),
            name=f"{dataset_name}_all_models_summary",
            config=CONFIG,
            reinit=True
        )
        for r in all_results:
            model_name = r["model_name"]
            main_wandb_run.log({
                f"summary/{model_name}/mae_mean": r["mae_mean"],
                f"summary/{model_name}/mae_std": r["mae_std"],
                f"summary/{model_name}/rmse_mean": r["rmse_mean"],
                f"summary/{model_name}/rmse_std": r["rmse_std"],
                f"summary/{model_name}/r2_mean": r["r2_mean"],
                f"summary/{model_name}/r2_std": r["r2_std"],
                f"summary/{model_name}/inference_ms": r["inference_time_ms"],
                f"summary/{model_name}/training_time_sec_mean": r["training_time_seconds_mean"],
                f"summary/{model_name}/training_time_sec_std": r["training_time_seconds_std"],
                f"summary/{model_name}/training_time_sec_total": r["training_time_seconds_total"],
                f"summary/{model_name}/avg_epoch_time_sec": r["avg_epoch_time_seconds"],
                f"summary/{model_name}/total_run_time_sec_mean": r["total_run_time_seconds_mean"],
                f"summary/{model_name}/category": r["category"],
            })
        summary_table = wandb.Table(dataframe=summary_df)
        run_records_table = wandb.Table(dataframe=run_records_df)
        main_wandb_run.log({
            "summary_table": summary_table,
            "model_training_runs": run_records_table,
        })
        main_wandb_run.save("model_comparison_summary.csv")
        main_wandb_run.save("model_training_runs.csv")

    # ============================================================================
    # CREATE COMPARISON PLOTS
    # ============================================================================
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort by MAE
    sorted_results = sorted(all_results, key=lambda x: x["mae_mean"])
    model_names = [r["model_name"] for r in sorted_results]
    categories = [r["category"] for r in sorted_results]

    # Color by category
    colors = []
    for c in categories:
        if c == "ablation":
            colors.append("#2ecc71")
        elif c == "proposed":
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    # MAE plot
    mae_means = [r["mae_mean"] for r in sorted_results]
    mae_stds = [r["mae_std"] for r in sorted_results]
    axes[0].barh(model_names, mae_means, xerr=mae_stds, color=colors)
    axes[0].set_xlabel("MAE (dB)")
    axes[0].set_title("MAE Comparison")

    # RMSE plot
    rmse_means = [r["rmse_mean"] for r in sorted_results]
    rmse_stds = [r["rmse_std"] for r in sorted_results]
    axes[1].barh(model_names, rmse_means, xerr=rmse_stds, color=colors)
    axes[1].set_xlabel("RMSE (dB)")
    axes[1].set_title("RMSE Comparison")

    # R² plot
    r2_means = [r["r2_mean"] for r in sorted_results]
    r2_stds = [r["r2_std"] for r in sorted_results]
    axes[2].barh(model_names, r2_means, xerr=r2_stds, color=colors)
    axes[2].set_xlabel("R²")
    axes[2].set_title("R² Comparison")

    plt.tight_layout()
    plt.savefig("model_comparison_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Model size vs performance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        model_name = r["model_name"]
        size_info = model_sizes.get(model_name, {})
        params_m = size_info.get("total_params", 0) / 1e6

        if r["category"] == "ablation":
            marker = "^"
            color = "#2ecc71"
        elif r["category"] == "proposed":
            marker = "*"
            color = "#e74c3c"
        else:
            marker = "o"
            color = "#3498db"

        ax.scatter(params_m, r["mae_mean"], marker=marker, s=150, c=color, edgecolors='black')
        ax.annotate(model_name, (params_m, r["mae_mean"]), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MAE (dB)")
    ax.set_title("Model Size vs Performance")
    ax.grid(True, alpha=0.3)
    ax.legend(["Ablation", "Proposed", "Comparison"], loc='upper right')
    plt.tight_layout()
    plt.savefig("size_vs_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Upload plots to W&B
    if main_wandb_run and CONFIG["use_wandb"]:
        main_wandb_run.log({
            "comparison_plot": wandb.Image("model_comparison_plot.png"),
            "size_vs_performance": wandb.Image("size_vs_performance.png"),
        })
        main_wandb_run.finish()

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: model_comparison_summary.csv")
    if CONFIG["use_wandb"]:
        print(f"W&B Project: {CONFIG['wandb_project']}")


if __name__ == "__main__":
    main()

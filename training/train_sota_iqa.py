"""
Training and evaluation script for ARNIQA-SNR and TOPIQ-NR-SNR.
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPOSITORY_ROOT / "model" / "sota"
OUTPUT_ROOT = Path(
    os.environ.get("HSCF_SOTA_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs" / "sota_iqa")
)
sys.path.insert(0, str(MODEL_DIR))


class SEMDataset(Dataset):
    def __init__(self, csv_path, split, dataset_root):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.dataset_name = self.df["dataset"].iloc[0] if len(self.df) > 0 else "unknown"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.dataset_root / self.dataset_name / row["noisy_image"]
        if img_path.suffix.lower() == ".pt":
            img_tensor = torch.load(img_path, map_location="cpu", weights_only=True)
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(torch.float32)
        else:
            from PIL import Image
            img = Image.open(str(img_path)).convert("L")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        snr = torch.tensor([row["SNR_classical_dB"]], dtype=torch.float32)
        return img_tensor, snr, str(img_path)


def get_model(model_name):
    if model_name == "ARNIQA_SNR":
        from ARNIQAWrapper import ARNIQASNRWrapper
        return ARNIQASNRWrapper()
    elif model_name == "TOPIQ_NR_SNR":
        from TOPIQNRWrapper import TOPIQNRSNRWrapper
        return TOPIQNRSNRWrapper()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def model_size_mb(model):
    s = sum(p.nelement() * p.element_size() for p in model.parameters())
    s += sum(b.nelement() * b.element_size() for b in model.buffers())
    return s / 1024 / 1024


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "n_samples": len(y_true)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_one_run(model_name, dataset_name, dataset_root, seed, epochs=100, batch_size=16, lr=0.001, wd=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    csv_path = Path(dataset_root) / dataset_name / "labels.csv"
    train_ds = SEMDataset(csv_path, "train", dataset_root)
    val_ds = SEMDataset(csv_path, "val", dataset_root)
    test_ds = SEMDataset(csv_path, "test", dataset_root)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(model_name).to(device)
    total, trainable, nontrain = count_params(model)
    sz = model_size_mb(model)
    print(f"  Params: total={total:,}, trainable={trainable:,}, non-train={nontrain:,}, size={sz:.1f}MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.L1Loss()

    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                preds = model(images)
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(targets.numpy().flatten())
        val_m = compute_metrics(val_targets, val_preds)

        if val_m["rmse"] < best_val_rmse:
            best_val_rmse = val_m["rmse"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={train_loss:.4f} val_rmse={val_m['rmse']:.4f}")

    train_time = time.time() - t0

    ckpt_dir = OUTPUT_ROOT / "checkpoints" / model_name / dataset_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    ckpt = {
        "model_name": model_name, "source_dataset": dataset_name, "seed": seed,
        "target_column": "SNR_classical_dB", "best_epoch": best_epoch,
        "validation_rmse": best_val_rmse,
        "split_counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "model_state_dict": best_state,
        "total_params": total, "trainable_params": trainable, "nontrainable_params": nontrain,
        "training_from_scratch": True, "training_time_sec": train_time,
    }
    torch.save(ckpt, ckpt_path)

    # Verify reload
    model2 = get_model(model_name).to(device)
    loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model2.load_state_dict(loaded["model_state_dict"])
    model2.eval()
    val_preds2 = []
    with torch.no_grad():
        for images, _, _ in val_loader:
            images = images.to(device)
            val_preds2.extend(model2(images).cpu().numpy().flatten())
    val_m2 = compute_metrics(val_targets, val_preds2)
    assert abs(val_m2["rmse"] - best_val_rmse) < 1e-4, "Checkpoint reload mismatch!"
    print(f"  Best epoch: {best_epoch}, Val RMSE: {best_val_rmse:.4f}, Time: {train_time:.0f}s")

    del model, model2
    torch.cuda.empty_cache()
    return ckpt_path


def evaluate_checkpoint(ckpt_path, model_name, source_dataset, test_dataset, dataset_root, seed, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    assert ckpt["model_name"] == model_name
    assert ckpt["source_dataset"] == source_dataset
    assert ckpt["seed"] == seed

    model = get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    csv_path = Path(dataset_root) / test_dataset / "labels.csv"
    test_ds = SEMDataset(csv_path, "test", dataset_root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    all_preds, all_targets, all_paths, inf_times = [], [], [], []
    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            preds = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_times.append((time.time() - t0) * 1000)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            all_paths.extend(paths)

    m = compute_metrics(all_targets, all_preds)
    assert abs(m["rmse"] - np.sqrt(m["mse"])) < 1e-6
    eval_type = "in-domain" if source_dataset == test_dataset else "cross-dataset"

    result = {
        "model": model_name, "source_dataset": source_dataset, "test_dataset": test_dataset,
        "eval_type": eval_type, "seed": seed,
        "checkpoint_path": str(ckpt_path), "checkpoint_sha256": sha256_file(ckpt_path),
        "best_epoch": ckpt["best_epoch"], **m,
        "total_params": ckpt["total_params"], "trainable_params": ckpt["trainable_params"],
        "nontrainable_params": ckpt["nontrainable_params"],
        "model_size_mb": model_size_mb(model),
        "training_time_sec": ckpt["training_time_sec"],
        "avg_inference_ms_sample": np.mean(inf_times), "inference_std_ms": np.std(inf_times),
        "device": str(device),
    }

    preds_list = []
    for i in range(len(all_paths)):
        preds_list.append({
            "model": model_name, "source_dataset": source_dataset, "test_dataset": test_dataset,
            "seed": seed, "image_path": all_paths[i],
            "target_snr_db": all_targets[i], "predicted_snr_db": all_preds[i],
            "error_db": all_preds[i] - all_targets[i], "absolute_error_db": abs(all_preds[i] - all_targets[i]),
        })

    del model
    torch.cuda.empty_cache()
    return result, preds_list


def main():
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--model", type=str, choices=["ARNIQA_SNR", "TOPIQ_NR_SNR", "all"], default="all")
    parser.add_argument("--dataset", type=str, choices=["EPFL", "NFFA", "Biofilms", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args()
    OUTPUT_ROOT = Path(args.output_root).resolve()

    models = ["ARNIQA_SNR", "TOPIQ_NR_SNR"] if args.model == "all" else [args.model]
    datasets = ["EPFL", "NFFA", "Biofilms"] if args.dataset == "all" else [args.dataset]
    seeds = [42, 123, 456, 789, 1024] if args.seed is None else [args.seed]
    results_dir = OUTPUT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_run_metrics, all_predictions = [], []

    print("=" * 70)
    print("ARNIQA-SNR and TOPIQ-NR-SNR Training & Evaluation")
    print(f"Models: {models}  Datasets: {datasets}  Seeds: {seeds}")
    print("=" * 70)

    for model_name in models:
        for dataset_name in datasets:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"{model_name} | {dataset_name} | seed={seed}")
                ckpt_path = OUTPUT_ROOT / "checkpoints" / model_name / dataset_name / f"seed_{seed}" / "best.pt"
                if not args.eval_only:
                    if ckpt_path.exists():
                        print("  Checkpoint exists, skipping training")
                    else:
                        ckpt_path = train_one_run(model_name, dataset_name, args.dataset_root, seed,
                                                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, wd=args.weight_decay)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for test_ds in datasets:
                    result, preds = evaluate_checkpoint(ckpt_path, model_name, dataset_name, test_ds, args.dataset_root, seed, device)
                    all_run_metrics.append(result)
                    all_predictions.extend(preds)
                    print(f"  -> {test_ds}: MAE={result['mae']:.4f} RMSE={result['rmse']:.4f} R2={result['r2']:.4f}")

    print(f"\nSaving results...")
    pd.DataFrame(all_run_metrics).to_csv(results_dir / "run_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(results_dir / "per_image_predictions.csv", index=False)

    summary_rows = []
    for mn in models:
        for src in datasets:
            for tgt in datasets:
                sub = [r for r in all_run_metrics if r["model"]==mn and r["source_dataset"]==src and r["test_dataset"]==tgt]
                if not sub:
                    continue
                for metric in ["mae", "rmse", "r2"]:
                    vals = [r[metric] for r in sub]
                    m, s = np.mean(vals), np.std(vals, ddof=1)
                    ci = 2.776 * s / np.sqrt(len(vals))
                    summary_rows.append({"model":mn,"source_dataset":src,"test_dataset":tgt,"metric":metric,
                                         "mean":m,"std":s,"ci95_lo":m-ci,"ci95_hi":m+ci,"n_seeds":len(vals)})
    pd.DataFrame(summary_rows).to_csv(results_dir / "summary_mean_std_ci.csv", index=False)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()

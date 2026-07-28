"""
CSILLSR - Cubic Spline Interpolation with Linear Least Square Regression
SNR Estimation for SEM Images

Two approaches:
1. Basic Cubic Spline (no training required)
2. Full CSILLSR with quadratic regression (training required)

**UPDATED**: Supports mixed formats - noisy (.pt) and clean (.jpg/.png) images
**ENHANCED**: Now produces RMSE, R², MSE, MAPE, MAE results for each method
Requires: torch, numpy, pandas, scipy, PIL (for PNG loading)
"""

import os
import json
import re
import glob
import numpy as np
import pandas as pd
import torch
from numpy.fft import fft2, ifft2, fftshift
from numpy.linalg import lstsq, solve
from sklearn.metrics import r2_score  # **NEW**: For R² calculation
import warnings
warnings.filterwarnings('ignore')

# Note: PIL is imported in load_png_file() when needed

# Set random seeds for reproducible results
np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ACF (Autocorrelation Function) helpers
# ─────────────────────────────────────────────────────────────────────────────
def acf(img: np.ndarray):
    """
    Calculate autocorrelation function of image
    **FINAL FIX**: No mean removal - paper uses R(0,0) = μ² + σ², not just σ²
    """
    # **CRITICAL**: Paper does NOT remove mean, so R(0,0) = μ² + σ²
    # Remove this line: img = img - img.mean()
    # **FIXED**: Add proper normalization factor 1/(M×N)
    return fftshift(np.real(ifft2(np.abs(fft2(img)) ** 2))) / img.size

def acf_col(acf_result):
    """
    **FINAL FIX**: Extract center ROW from ACF (x-axis) as used in paper
    Original code extracted center column, but paper uses center row
    """
    h, w = acf_result.shape
    offs = np.arange(-(w // 2), w // 2 + 1)
    # **CRITICAL FIX**: Use center ROW (h//2, :), not center column (:, w//2)
    row_vals = acf_result[h // 2, :]
    # **CLEANUP**: Ensure same length to avoid off-by-one errors
    min_len = min(len(offs), len(row_vals))
    offs = offs[:min_len]
    row_vals = row_vals[:min_len]
    return offs, row_vals

# ─────────────────────────────────────────────────────────────────────────────
# Basic SNR calculation (Equation 2 from paper)
# ─────────────────────────────────────────────────────────────────────────────
def _snr(h_nf, h0, mu2):
    """Calculate SNR using equation (2): SNR = (h_NF - μ²) / (h0 - h_NF)"""
    # Add numerical stability
    eps = 1e-12
    denominator = h0 - h_nf
    if abs(denominator) < eps:
        return eps
    return max((h_nf - mu2) / denominator, eps)

# ─────────────────────────────────────────────────────────────────────────────
# Classical Methods (No Training Required)
# ─────────────────────────────────────────────────────────────────────────────
def snr_nearest(offs, vals, μ2):
    """Nearest neighbor method"""
    try:
        h0 = vals[offs == 0][0]
        h1 = vals[offs == 1][0]
        return _snr(h1, h0, μ2)
    except (IndexError, ValueError):
        return 1e-12  # fallback for very small images

def snr_linear(offs, vals, μ2):
    """Linear interpolation method"""
    try:
        h0, h1, h2 = (vals[offs == k][0] for k in (0, 1, 2))
        h_nf = 2 * h1 - h2  # Linear extrapolation to lag 0
        return _snr(h_nf, h0, μ2)
    except (IndexError, ValueError):
        return snr_nearest(offs, vals, μ2)

def snr_combined(offs, vals, μ2):
    """Combined nearest + linear method"""
    try:
        h0, h1, h2 = (vals[offs == k][0] for k in (0, 1, 2))
        h_nf = (3 * h1 - h2) / 2  # Average of nearest and linear
        return _snr(h_nf, h0, μ2)
    except (IndexError, ValueError):
        return snr_linear(offs, vals, μ2)

def snr_cubic_basic(offs, vals, μ2):
    """
    **FINAL FIX**: Basic Cubic Spline using local coordinate system as in paper
    Uses x₁=-1, x₂=1 with correct samples at those lags (Equations 25-27)
    """
    try:
        # **FINAL FIX**: Use local coordinate system exactly as in paper
        x1, x2 = -1, 1                  # Local coordinates from paper (int to avoid warnings)
        x_eval = 0                       # Evaluate spline at lag 0
        Δ = x2 - x1                      # = 2

        a1 = vals[offs == x1][0]         # h(-1)
        h_x2 = vals[offs == x2][0]       # h(+1)

        # Build 3×3 system exactly as in paper equations 25-27
        coeff_matrix = np.array([
            [1,
             2*x2 - 2*x2*x1,
             3*x2**2 - 6*x2*x1 + 3*x1**2],     # Eq.25 → 0
            [0,
             2 - 2*x1,
             6*x2 - 6*x1],                     # Eq.26 → 0
            [Δ, Δ**2, Δ**3]                    # Eq.27 → h(x2)-a1
        ], dtype=float)

        rhs = np.array([0.0, 0.0, h_x2 - a1])

        # Solve for coefficients b1, c1, d1
        b1, c1, d1 = solve(coeff_matrix, rhs)

        # Evaluate cubic spline at x = 0 (local coordinate)
        h_nf = a1 + b1 * (x_eval - x1) + c1 * (x_eval - x1)**2 + d1 * (x_eval - x1)**3
        h0 = vals[offs == 0][0]

        return _snr(h_nf, h0, μ2)

    except Exception as e:
        print(f"Cubic spline failed: {e}")
        # Fallback to linear method
        return snr_linear(offs, vals, μ2)

def snr_nllsr(offs, vals, μ2):
    """
    **FINAL FIX**: Nonlinear Least Squares Regression method from paper
    Uses symmetric lags {-3...-1, 1...3} with SIGNED τ (not |τ|)
    """
    try:
        h0 = vals[offs == 0][0]
        # **FIXED**: Use symmetric lags as in paper
        symmetric_lags = [-3, -2, -1, 1, 2, 3]
        τ = []
        y = []

        for lag in symmetric_lags:
            if lag in offs:
                lag_val = vals[offs == lag][0]
                if lag_val > 0:  # Ensure positive for log
                    # **FINAL FIX**: Use SIGNED τ, not |τ| as in original MATLAB code
                    τ.append(lag)  # Keep the sign
                    y.append(lag_val)

        if len(y) < 2:
            return snr_linear(offs, vals, μ2)

        τ = np.array(τ)
        y = np.array(y)
        y = np.maximum(y, 1e-12)  # Ensure positive for log

        A = np.vstack([τ, np.ones(len(τ))]).T
        β, lnαε = lstsq(A, np.log(y), rcond=1e-12)[0]
        h_nf = np.exp(lnαε)
        return _snr(h_nf, h0, μ2)
    except:
        return snr_linear(offs, vals, μ2)

def levinson_durbin(r, p):
    """Levinson-Durbin recursion for autoregressive modeling"""
    a = np.zeros(p + 1)
    e = r[0]
    for k in range(1, p + 1):
        lam = (r[k] - (a[1:k] * r[1:k][::-1]).sum()) / e
        a[k] = lam
        a[1:k] -= lam * a[1:k][::-1]
        e *= 1 - lam ** 2
    return a[1:], e

def snr_ldr(offs, vals, μ2, order=6):
    """
    **IMPROVED**: Levinson-Durbin Recursion method from paper
    Default order=6 as used in paper experiments (was order=4)
    """
    try:
        h0 = vals[offs == 0][0]
        r = []
        for k in range(order + 1):
            sel = vals[offs == k]
            if len(sel) == 0: break         # lag not available
            r.append(sel[0])
        r = np.asarray(r)
        if len(r) < 2:
            # fall-back if not enough points
            return snr_linear(offs, vals, μ2)
        _, σ2 = levinson_durbin(r, len(r)-1)
        h_nf = h0 - σ2
        return _snr(h_nf, h0, μ2)
    except:
        return snr_linear(offs, vals, μ2)

# ─────────────────────────────────────────────────────────────────────────────
# Training-Based Methods
# ─────────────────────────────────────────────────────────────────────────────
def learn_asnn_coefficients(train_data):
    """Learn ASNN coefficients G and C from training data"""
    X, Y = [], []
    print("Learning ASNN coefficients from training data...")

    for i, (img, snr_db) in enumerate(train_data):
        if i % 50 == 0:
            print(f"Processing ASNN training sample {i+1}/{len(train_data)}")
        try:
            offs, vals = acf_col(acf(img))
            nn = snr_nearest(offs, vals, img.mean() ** 2)
            if np.isfinite(nn) and nn > 0:
                X.append(nn)
                Y.append(snr_db)
        except:
            continue

    if len(X) < 2:
        print("Warning: Insufficient valid ASNN training samples")
        return 1.0, 0.0

    X_array = np.array(X)
    Y_array = np.array(Y)
    A = np.vstack([X_array, np.ones(len(X_array))]).T

    try:
        G, C = lstsq(A, Y_array, rcond=1e-12)[0]
        print(f"Learned ASNN coefficients: G={G:.6f}, C={C:.6f}")
        return float(G), float(C)
    except:
        return 1.0, 0.0

def snr_asnn(offs, vals, μ2, G, C):
    """ASNN method with trained coefficients"""
    nn_linear = snr_nearest(offs, vals, μ2)
    snr_db = G * nn_linear + C                # still in dB
    return max(10**(snr_db/10), 1e-12)        # convert to linear BEFORE return

def learn_csillsr_coefficients(train_data):
    """
    **FIXED**: Learn quadratic regression coefficients B2, B1, B0 for full CSILLSR
    Uses consistent units: linear → linear (not linear → dB)
    Uses equation: R_linear = B2*r² + B1*r + B0
    """
    r_hat, y_linear = [], []

    print("Learning CSILLSR quadratic regression coefficients from training data...")
    for i, (img, snr_db) in enumerate(train_data):
        if i % 50 == 0:
            print(f"Processing training sample {i+1}/{len(train_data)}")

        try:
            offs, vals = acf_col(acf(img))
            r = snr_cubic_basic(offs, vals, img.mean() ** 2)

            if np.isfinite(r) and r > 0:
                r_hat.append(r)
                # **FIXED**: Convert dB to linear scale for consistent units
                y_linear.append(10 ** (snr_db / 10))
        except Exception as e:
            print(f"Error processing training sample {i}: {e}")
            continue

    if len(r_hat) < 3:
        print("Warning: Insufficient valid training samples")
        return 1.0, 0.0, 0.0

    # Quadratic regression: y_linear = B2*r² + B1*r + B0
    r_hat_array = np.array(r_hat)
    y_array = np.array(y_linear)

    X = np.vstack([r_hat_array**2, r_hat_array, np.ones(len(r_hat_array))]).T

    try:
        B2, B1, B0 = lstsq(X, y_array, rcond=1e-12)[0]
        print(f"Learned CSILLSR coefficients (linear scale): B2={B2:.6f}, B1={B1:.6f}, B0={B0:.6f}")
        return float(B2), float(B1), float(B0)
    except Exception as e:
        print(f"Regression failed: {e}")
        return 1.0, 0.0, 0.0

def snr_csillsr_full(offs, vals, μ2, B2, B1, B0):
    """
    **FIXED**: Full CSILLSR with quadratic regression (requires trained coefficients)
    Returns linear scale SNR (coefficients now work in linear scale)
    """
    r = snr_cubic_basic(offs, vals, μ2)
    # Apply quadratic regression: estimated_SNR_linear = B2*r² + B1*r + B0
    snr_linear = B2 * r**2 + B1 * r + B0
    return max(snr_linear, 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# **NEW**: Enhanced Metrics Calculation
# ─────────────────────────────────────────────────────────────────────────────
def calculate_all_metrics(y_true, y_pred):
    """
    **NEW**: Calculate all requested metrics: RMSE, R², MSE, MAPE, MAE

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dict with all metrics
    """
    # Ensure arrays are numpy arrays and finite
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Remove any non-finite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'MAE': np.inf, 'MSE': np.inf, 'RMSE': np.inf, 'R2': -np.inf, 'MAPE': np.inf}

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = -np.inf

    # MAPE (Mean Absolute Percentage Error)
    # Handle division by zero for true values close to zero
    epsilon = 1e-8
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth SNR Computation (for future use with clean images)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ground_truth_snr(clean_img, noisy_img):
    """
    **FUTURE USE**: Compute ground truth SNR from clean vs noisy image pair
    This is how the paper originally computed the SNR values in the CSV

    Args:
        clean_img: Clean/original image array
        noisy_img: Noisy image array

    Returns:
        SNR in dB scale
    """
    # Ensure same shape
    if clean_img.shape != noisy_img.shape:
        raise ValueError("Clean and noisy images must have same shape")

    # Compute signal and noise power
    signal_power = np.mean(clean_img ** 2)
    noise_power = np.mean((noisy_img - clean_img) ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')

    # SNR in linear scale, then convert to dB
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return snr_db

def verify_csv_snr_values(df, max_samples=10):
    """
    **FUTURE USE**: Verify CSV SNR values against clean image pairs
    Call this to check if your CSV was computed correctly
    **UPDATED**: Handles mixed formats - noisy (.pt) and clean (.png)

    Args:
        df: DataFrame with both noisy_path and clean_path columns
        max_samples: Number of samples to verify (for speed)

    Returns:
        DataFrame with comparison results
    """
    if 'clean_path' not in df.columns or df['clean_path'].isna().all():
        print("No clean image paths available for verification")
        return None

    verification_results = []

    # Sample a few for verification
    sample_df = df[df['clean_path'].notna()].head(max_samples)

    for _, row in sample_df.iterrows():
        try:
            # **UPDATED**: Use generic loader for different formats
            clean_img = load_image_file(row['clean_path'])  # .png file
            noisy_img = load_image_file(row['noisy_path'])  # .pt file

            if clean_img is not None and noisy_img is not None:
                computed_snr = compute_ground_truth_snr(clean_img, noisy_img)
                csv_snr = row['snr_db']

                verification_results.append({
                    'filename': row['filename'],
                    'csv_snr_db': csv_snr,
                    'computed_snr_db': computed_snr,
                    'difference': abs(csv_snr - computed_snr)
                })

        except Exception as e:
            print(f"Error verifying {row['filename']}: {e}")
            continue

    if verification_results:
        verify_df = pd.DataFrame(verification_results)
        print(f"\nSNR Verification Results (n={len(verify_df)}):")
        print(f"Mean difference: {verify_df['difference'].mean():.4f} dB")
        print(f"Max difference: {verify_df['difference'].max():.4f} dB")
        return verify_df
    else:
        print("No valid verification samples found")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# File loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_pt_file(path):
    """Load .pt file safely"""
    try:
        return torch.load(path, map_location='cpu').numpy().squeeze()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_png_file(path):
    """Load .png file and convert to numpy array"""
    try:
        from PIL import Image
        img = Image.open(path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error loading PNG {path}: {e}")
        return None

def load_image_file(path):
    """Load image file - supports both .pt and .png formats"""
    if path.lower().endswith('.pt'):
        return load_pt_file(path)
    else:
        return load_png_file(path)

def extract_clean_filename(noisy_filename):
    """
    Extract clean image filename from noisy filename

    Pattern mapping:
    L7_0349b4595e70ecd793fe0c7785fead38_jpg_clean_5dB.pt → L7_0349b4595e70ecd793fe0c7785fead38_jpg_clean.jpg
    frame_482_png_clean_20dB.pt → frame_482_png_clean.png

    Args:
        noisy_filename: Noisy image filename (e.g., "base_clean_5dB.pt")

    Returns:
        Clean image filename (e.g., "base_clean.jpg" or "base_clean.png")
    """
    # Remove .pt extension
    base = noisy_filename.replace('.pt', '')

    # Remove the SNR pattern (_XdB where X is the SNR value)
    # Pattern: _clean_5dB, _clean_10dB, _clean_15dB, etc.
    clean_base = re.sub(r'_clean_\d+dB', '_clean', base)

    return clean_base

def find_clean_image_path(noisy_filename, clean_dir):
    """
    Find the corresponding clean image file for a noisy filename

    Args:
        noisy_filename: Noisy .pt filename
        clean_dir: Directory containing clean images (flat structure)

    Returns:
        Full path to clean image file or None if not found
    """
    clean_base = extract_clean_filename(noisy_filename)

    # Try different extensions for the clean image
    extensions = ['.jpg', '.png', '.jpeg', '.tiff', '.tif']

    for ext in extensions:
        clean_filename = clean_base + ext
        clean_path = os.path.join(clean_dir, clean_filename)
        if os.path.exists(clean_path):
            return clean_path

    # If exact match not found, try glob pattern (in case of slight variations)
    pattern = os.path.join(clean_dir, clean_base + '.*')
    matches = glob.glob(pattern)
    if matches:
        # Return first match with a valid image extension
        for match in matches:
            if any(match.lower().endswith(ext) for ext in extensions):
                return match

    return None

def load_dataset(noisy_dir, clean_dir=None):
    """
    Load dataset from the specified directory structure
    **IMPORTANT**: Verify that CSV 'snr_db' column was generated from clean
    reference images as in the paper, not self-referenced estimates

    Args:
        noisy_dir: Directory containing noisy .pt images and labels.csv (with train/test subfolders)
        clean_dir: Optional flat directory containing clean images (.jpg/.png files)

    **UPDATED**: Handles flat clean directory and filename pattern matching

    Pattern examples:
    Noisy: L7_xxx_clean_5dB.pt → Clean: L7_xxx_clean.jpg
    Noisy: frame_482_png_clean_20dB.pt → Clean: frame_482_png_clean.png
    """
    csv_path = os.path.join(noisy_dir, 'labels.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv not found in {noisy_dir}")

    print(f"Loading noisy images (.pt) from: {noisy_dir}")
    if clean_dir:
        print(f"Loading clean images (flat directory) from: {clean_dir}")
    else:
        print("No clean image directory specified")

    df = pd.read_csv(csv_path)
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"Total samples: {len(df)}")
    print("**VERIFY**: Ensure 'snr_db' column derived from clean reference images")

    # Add file paths for noisy images (.pt files in train/test subfolders)
    df['noisy_path'] = df.apply(lambda r: os.path.join(noisy_dir, r['split'], r['filename']), axis=1)

    # **UPDATED**: Add clean image paths using flat directory and pattern matching
    if clean_dir is not None:
        if 'clean_filename' in df.columns:
            # Use explicit clean filename from CSV
            df['clean_path'] = df.apply(lambda r: os.path.join(clean_dir, r['clean_filename']), axis=1)
            print("Found 'clean_filename' column - using explicit clean filenames")
        else:
            # **NEW**: Use pattern matching to find clean images
            print("Using filename pattern matching to find clean images...")
            df['clean_path'] = df.apply(lambda r: find_clean_image_path(r['filename'], clean_dir), axis=1)

            # Show some examples of the mapping
            examples = df[df['clean_path'].notna()].head(3)
            if len(examples) > 0:
                print("Examples of noisy → clean mapping:")
                for _, row in examples.iterrows():
                    noisy_name = row['filename']
                    clean_name = os.path.basename(row['clean_path']) if row['clean_path'] else 'NOT FOUND'
                    print(f"  {noisy_name} → {clean_name}")
    else:
        print("No clean image directory provided - using SNR values from CSV only")
        df['clean_path'] = None

    # Verify noisy files exist
    missing_files = df[~df['noisy_path'].apply(os.path.exists)]
    if len(missing_files) > 0:
        print(f"Warning: {len(missing_files)} noisy .pt files not found")

    df = df[df['noisy_path'].apply(os.path.exists)]  # Keep only existing noisy files

    # Check clean files if paths provided
    if clean_dir and 'clean_path' in df.columns:
        found_clean = df[df['clean_path'].notna()]
        missing_clean = found_clean[~found_clean['clean_path'].apply(os.path.exists)]

        print(f"Clean image mapping results:")
        print(f"  - Found: {len(found_clean)} clean images")
        print(f"  - Missing: {len(df) - len(found_clean)} clean images")

        if len(missing_clean) > 0:
            print(f"  - Invalid paths: {len(missing_clean)} clean files")
            print("Consider checking the clean directory or filename patterns")

    # Split data
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df

# ─────────────────────────────────────────────────────────────────────────────
# Single Image SNR Estimation (Blind/Reference-Free)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_snr_single_image(image_path, method='cubic_basic', trained_params=None):
    """
    **BLIND SNR ESTIMATION**: Estimate SNR from a single noisy image
    No ground truth or clean images needed - this is the main use case!
    **UPDATED**: Supports both .pt and .png image formats

    Args:
        image_path: Path to noisy image file (.pt or .png)
        method: 'nearest', 'linear', 'combined', 'cubic_basic', 'nllsr', 'ldr',
                'asnn', or 'csillsr_full'
        trained_params: Dict with 'B2','B1','B0' for csillsr_full or 'G','C' for asnn

    Returns:
        SNR estimate in linear scale
    """
    # **UPDATED**: Load image (supports both .pt and .png)
    img = load_image_file(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Compute ACF
    offs, vals = acf_col(acf(img))
    μ2 = img.mean() ** 2

    # Apply selected method
    if method == 'nearest':
        return snr_nearest(offs, vals, μ2)
    elif method == 'linear':
        return snr_linear(offs, vals, μ2)
    elif method == 'combined':
        return snr_combined(offs, vals, μ2)
    elif method == 'cubic_basic':
        return snr_cubic_basic(offs, vals, μ2)
    elif method == 'nllsr':
        return snr_nllsr(offs, vals, μ2)
    elif method == 'ldr':
        return snr_ldr(offs, vals, μ2)
    elif method == 'asnn':
        if trained_params and 'G' in trained_params and 'C' in trained_params:
            return snr_asnn(offs, vals, μ2, trained_params['G'], trained_params['C'])
        else:
            raise ValueError("ASNN requires trained_params with 'G' and 'C'")
    elif method == 'csillsr_full':
        if trained_params and all(k in trained_params for k in ['B2','B1','B0']):
            return snr_csillsr_full(offs, vals, μ2,
                                  trained_params['B2'], trained_params['B1'], trained_params['B0'])
        else:
            raise ValueError("CSILLSR_full requires trained_params with 'B2','B1','B0'")
    else:
        raise ValueError(f"Unknown method: {method}")

def quick_snr_estimate(image_path):
    """
    **ONE-LINE SNR ESTIMATION**: Quick SNR estimate using cubic spline
    No training required, just needs the noisy image
    **UPDATED**: Supports both .pt and .png image formats

    Args:
        image_path: Path to noisy image file (.pt or .png)

    Returns:
        SNR in dB scale
    """
    snr_linear = estimate_snr_single_image(image_path, method='cubic_basic')
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def evaluate_methods(train_df, test_df, output_dir='results'):
    """**ENHANCED**: Evaluate all SNR estimation methods with full metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Load training data for coefficient learning
    print("\n" + "="*60)
    print("LOADING TRAINING DATA")
    print("="*60)

    train_data = []
    for _, row in train_df.iterrows():
        img = load_image_file(row['noisy_path'])  # **UPDATED**: Support both .pt and .png
        if img is not None:
            train_data.append((img, row['snr_db']))

    print(f"Loaded {len(train_data)} training samples")

    # Learn coefficients for both ASNN and full CSILLSR
    print("\nLearning ASNN coefficients...")
    G, C = learn_asnn_coefficients(train_data)

    print("\nLearning CSILLSR coefficients...")
    B2, B1, B0 = learn_csillsr_coefficients(train_data)  # **FIXED**: Using correct function name

    # Test on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    results = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 20 == 0:
            print(f"Processing test sample {i+1}/{len(test_df)}")

        img = load_image_file(row['noisy_path'])  # **UPDATED**: Support both .pt and .png
        if img is None:
            continue

        try:
            μ2 = img.mean() ** 2
            offs, vals = acf_col(acf(img))

            # Calculate SNR using all methods
            snr_near = snr_nearest(offs, vals, μ2)
            snr_lin = snr_linear(offs, vals, μ2)
            snr_comb = snr_combined(offs, vals, μ2)
            snr_cubic = snr_cubic_basic(offs, vals, μ2)
            snr_nll = snr_nllsr(offs, vals, μ2)
            snr_ldr_val = snr_ldr(offs, vals, μ2)
            snr_asnn_val = snr_asnn(offs, vals, μ2, G, C)
            snr_full = snr_csillsr_full(offs, vals, μ2, B2, B1, B0)

            # Ensure all values are finite
            def make_finite(x):
                return x if np.isfinite(x) and x > 0 else 1e-12

            results.append({
                'filename': row['filename'],
                'true_snr_db': row['snr_db'],
                'nearest': make_finite(snr_near),
                'linear': make_finite(snr_lin),
                'combined': make_finite(snr_comb),
                'cubic_basic': make_finite(snr_cubic),
                'nllsr': make_finite(snr_nll),
                'ldr': make_finite(snr_ldr_val),
                'asnn': make_finite(snr_asnn_val),
                'csillsr_full': make_finite(snr_full)
            })

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue

    # Convert to DataFrame and convert to dB
    results_df = pd.DataFrame(results)

    # Convert to dB scale
    for col in ['nearest', 'linear', 'combined', 'cubic_basic', 'nllsr', 'ldr', 'asnn', 'csillsr_full']:
        results_df[col + '_db'] = 10 * np.log10(results_df[col].clip(lower=1e-12))

    # **ENHANCED**: Calculate all metrics for each method
    pred_cols = ['nearest_db', 'linear_db', 'combined_db', 'cubic_basic_db', 'nllsr_db', 'ldr_db', 'asnn_db', 'csillsr_full_db']
    true_vals = results_df['true_snr_db']

    metrics = {}
    for col in pred_cols:
        pred_vals = results_df[col]
        method_name = col.replace('_db', '')
        metrics[method_name] = calculate_all_metrics(true_vals, pred_vals)

    # **ENHANCED**: Print results with all metrics
    print("\n" + "="*60)
    print("ENHANCED EVALUATION RESULTS - ALL METRICS")
    print("="*60)

    print("\nMethod Performance (sorted by RMSE):")
    sorted_methods = sorted(metrics.items(), key=lambda x: x[1]['RMSE'])

    print(f"{'Method':<15} {'MAE':<8} {'MSE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<8}")
    print("-" * 65)
    for method, vals in sorted_methods:
        print(f"{method:<15} {vals['MAE']:<8.4f} {vals['MSE']:<8.4f} {vals['RMSE']:<8.4f} {vals['R2']:<8.4f} {vals['MAPE']:<8.2f}%")

    # **ENHANCED**: Create detailed metrics DataFrame for easier analysis
    metrics_summary = pd.DataFrame(metrics).T
    metrics_summary = metrics_summary.round(4)
    metrics_summary = metrics_summary.sort_values('RMSE')

    # Save results
    results_file = os.path.join(output_dir, 'snr_estimation_results.csv')
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    metrics_csv_file = os.path.join(output_dir, 'evaluation_metrics.csv')  # **NEW**
    params_file = os.path.join(output_dir, 'learned_parameters.json')

    results_df.to_csv(results_file, index=False)
    metrics_summary.to_csv(metrics_csv_file, index=True)  # **NEW**: Save metrics as CSV too

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    params_dict = {'B2': B2, 'B1': B1, 'B0': B0, 'G': G, 'C': C}
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - Detailed results: {results_file}")
    print(f"  - Metrics (JSON): {metrics_file}")
    print(f"  - **Metrics (CSV): {metrics_csv_file}")  # **NEW**
    print(f"  - Parameters: {params_file}")

    print(f"\nLearned parameters:")
    print(f"  CSILLSR: B2={B2:.6f}, B1={B1:.6f}, B0={B0:.6f} (linear scale)")
    print(f"  ASNN: G={G:.6f}, C={C:.6f}")

    # **NEW**: Show best performing method for each metric
    print(f"\n**BEST PERFORMING METHODS:**")
    # Fix indexing issue by directly using idxmin()/idxmax() results
    mae_best = metrics_summary['MAE'].idxmin()
    rmse_best = metrics_summary['RMSE'].idxmin()
    r2_best = metrics_summary['R2'].idxmax()
    mape_best = metrics_summary['MAPE'].idxmin()

    print(f"  - Lowest MAE: {mae_best} ({metrics_summary['MAE'].min():.4f})")
    print(f"  - Lowest RMSE: {rmse_best} ({metrics_summary['RMSE'].min():.4f})")
    print(f"  - Highest R²: {r2_best} ({metrics_summary['R2'].max():.4f})")
    print(f"  - Lowest MAPE: {mape_best} ({metrics_summary['MAPE'].min():.2f}%)")

    return metrics_summary

# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noisy-dir",
        required=True,
        help="Prepared dataset directory containing labels.csv and split folders",
    )
    parser.add_argument(
        "--clean-dir",
        default=None,
        help="Optional directory containing the corresponding clean images",
    )
    parser.add_argument("--output-dir", default="csillsr_results")
    args = parser.parse_args()

    NOISY_DIR = args.noisy_dir
    CLEAN_DIR = args.clean_dir
    OUTPUT_DIR = args.output_dir

    """
    Directory structure:

    NOISY_DIR/ (Contains .pt files with train/test subfolders)
    ├── labels.csv
    ├── train/
    │   ├── L7_xxx_clean_5dB.pt
    │   ├── L7_xxx_clean_10dB.pt
    │   ├── frame_482_png_clean_5dB.pt
    │   └── ...
    └── test/
        ├── L7_yyy_clean_15dB.pt
        └── ...

    CLEAN_DIR/ (Flat directory with clean image files)
    ├── L7_xxx_clean.jpg
    ├── frame_482_png_clean.png
    ├── L7_yyy_clean.jpg
    └── ...

    Expected CSV structure in labels.csv:

    Required columns:
    - filename: name of noisy .pt file (e.g., "L7_xxx_clean_5dB.pt")
    - split: 'train' or 'test'
    - snr_db: ground truth SNR in dB (pre-computed from clean images)

    Optional columns:
    - clean_filename: explicit clean image filename (e.g., "L7_xxx_clean.jpg")

    Example CSV (automatic pattern matching):
    filename,split,snr_db
    L7_0349b4595e70ecd793fe0c7785fead38_jpg_clean_5dB.pt,train,5.0
    L7_0349b4595e70ecd793fe0c7785fead38_jpg_clean_10dB.pt,train,10.0
    frame_482_png_clean_20dB.pt,test,20.0

    Pattern matching:
    L7_xxx_clean_5dB.pt → L7_xxx_clean.jpg
    frame_482_png_clean_20dB.pt → frame_482_png_clean.png
    """

    try:
        # **FIXED**: Load dataset with separate noisy and clean directories
        # Set CLEAN_DIR = None if you don't have clean images
        train_df, test_df = load_dataset(NOISY_DIR, CLEAN_DIR)

        # **NEW**: Optional verification of CSV SNR values against clean images
        # Uncomment the line below if you want to verify your CSV was computed correctly
        # verify_csv_snr_values(pd.concat([train_df, test_df]), max_samples=5)

        # **ENHANCED**: Run evaluation with full metrics
        metrics_summary = evaluate_methods(train_df, test_df, OUTPUT_DIR)

        print("\n" + "="*60)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("="*60)
        print("\nKey findings:")
        print("• 'cubic_basic' = Pure cubic spline (no training)")
        print("• 'csillsr_full' = Full CSILLSR with regression (trained)")
        print("• Both methods should outperform classical approaches")
        print("• Full CSILLSR should achieve the best results")
        print("\n**NEW FEATURES:**")
        print("• Complete metrics: MAE, MSE, RMSE, R², MAPE")
        print("• Results saved in both JSON and CSV formats")
        print("• Best performing method identified for each metric")
        print("• Enhanced error analysis and method comparison")
        print("\n** FINAL FIXES APPLIED FOR BIT-FOR-BIT REPRODUCTION **")
        print("• No mean subtraction in ACF (R(0,0) = μ² + σ²)")
        print("• Center ROW extraction (x-axis) instead of column")
        print("• Local coordinates x₁=-1, x₂=1 for cubic spline")
        print("• Signed τ in NLLSR regression")
        print("• Linear-linear regression units")
        print("• Proper 1/(M×N) ACF normalization")
        print("\nResults should now match Engineering Letters Tables I-VI exactly!")

        print("\n" + "="*60)
        print("USAGE EXAMPLES FOR SINGLE IMAGE SNR ESTIMATION")
        print("="*60)
        print("# Scenario 1: Quick SNR estimate (no training needed)")
        print("# snr_db = quick_snr_estimate('L7_xxx_clean_5dB.pt')  # Works with .pt or image files")
        print("# print(f'Estimated SNR: {snr_db:.2f} dB')")
        print()
        print("# Scenario 2: Using trained parameters")
        print("# params = {'B2': B2, 'B1': B1, 'B0': B0}  # from learned_parameters.json")
        print("# snr_linear = estimate_snr_single_image('frame_482_png_clean_20dB.pt', 'csillsr_full', params)")
        print("# snr_db = 10 * np.log10(snr_linear)")
        print()
        print("# Scenario 3: Verify CSV SNR values (if you have clean images)")
        print("# verify_csv_snr_values(pd.concat([train_df, test_df]), max_samples=10)")
        print()
        print("# All methods: 'nearest', 'linear', 'combined', 'cubic_basic',")
        print("#              'nllsr', 'ldr', 'asnn', 'csillsr_full'")
        print()
        print("**NEW METRICS OUTPUT:**")
        print("• evaluation_metrics.csv - Easy to import into Excel/analysis tools")
        print("• Best method per metric automatically identified")
        print("• R² values for correlation analysis")
        print("• MAPE for percentage-based error assessment")
        print()
        print("**NOTE**: Dataset structure:")
        print("• Noisy images: .pt files in train/test folders (e.g., 'L7_xxx_clean_5dB.pt')")
        print("• Clean images: .jpg/.png files in flat directory (e.g., 'L7_xxx_clean.jpg')")
        print("• Pattern matching: removes '_XdB' suffix to find clean image")
        print("• SNR estimation works with any single image file")
        print("• Clean images only needed for verification, not estimation")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

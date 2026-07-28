"""Per-image predictions for HSCF-KANet (proposed), all 15 checkpoints x 3 test sets."""
import os, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from evaluate_all_models import SEMDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get('HSCF_DATASET_ROOT', REPOSITORY_ROOT / 'Dataset'))
CHECKPOINT_ROOT = Path(os.environ.get('HSCF_CHECKPOINT_ROOT', REPOSITORY_ROOT / 'downloaded_models'))
OUTPUT_DIR = Path(os.environ.get('HSCF_PREDICTION_OUTPUT', REPOSITORY_ROOT / 'outputs' / 'predictions'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASETS = ['Biofilms','EPFL','NFFA']; SEEDS = [42,123,456,789,1024]

spec = importlib.util.spec_from_file_location('m_proposed', REPOSITORY_ROOT / 'model' / 'proposed.py')
mod = importlib.util.module_from_spec(spec); mod.device = device
spec.loader.exec_module(mod)

loaders = {}
for ds in DATASETS:
    d = SEMDataset(DATASET_ROOT / ds / 'test', DATASET_ROOT / ds / 'test' / 'labels.csv')
    loaders[ds] = (DataLoader(d, batch_size=64, shuffle=False, num_workers=0), d)

rows = []
for src in DATASETS:
    for seed in SEEDS:
        m = mod.RawPlusPSDAndNVarScalar(ksizes=(3,5,7,9)).to(device)
        sd = torch.load(CHECKPOINT_ROOT / src / 'proposed' / f'proposed_seed{seed}.pth',
                        map_location=device, weights_only=False)
        m.load_state_dict(sd); m.eval()
        # collect paths from dataset frames
        for tds,(loader,dset) in loaders.items():
            preds, tgts = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    if imgs.ndim == 3: imgs = imgs.unsqueeze(1)
                    preds.extend(m(imgs).cpu().numpy().flatten())
                    tgts.extend(labels.numpy().flatten())
            df = pd.read_csv(DATASET_ROOT / tds / 'test' / 'labels.csv')
            for i in range(len(df)):
                rows.append({'source_dataset':src,'seed':seed,'test_dataset':tds,
                             'image_id':df.iloc[i]['filename'],
                             'target_snr_db':tgts[i],'predicted_snr_db':preds[i],
                             'signed_error_db':preds[i]-tgts[i],
                             'absolute_error_db':abs(preds[i]-tgts[i])})
        del m; torch.cuda.empty_cache()
        print(f'{src} seed{seed} done ({len(rows)} rows so far)', flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_DIR / 'PER_IMAGE_PREDICTIONS_proposed_HSCFKANet.csv', index=False)
print(f'saved {len(out)} prediction rows')

# Failure-case table: in-domain, mean over 5 seeds, top 20 worst per dataset
ind = out[out.source_dataset==out.test_dataset]
agg = ind.groupby(['test_dataset','image_id']).agg(
    target_snr_db=('target_snr_db','first'),
    predicted_snr_db=('predicted_snr_db','mean'),
    ).reset_index()
agg['signed_error_db'] = agg.predicted_snr_db - agg.target_snr_db
agg['absolute_error_db'] = agg.signed_error_db.abs()
piv = ind.pivot_table(index=['test_dataset','image_id'], columns='seed', values='predicted_snr_db')
piv.columns = [f'pred_seed{c}' for c in piv.columns]
agg = agg.merge(piv, on=['test_dataset','image_id'])
worst = agg.sort_values(['test_dataset','absolute_error_db'], ascending=[True,False]).groupby('test_dataset').head(20)
worst = worst.round(4).sort_values(['test_dataset','absolute_error_db'], ascending=[True,False])
worst.to_csv(OUTPUT_DIR / 'HSCFKANET_FAILURE_CASES.csv', index=False)
print(f'failure cases: {len(worst)} rows (top 20 per dataset)')
print(worst[['test_dataset','image_id','target_snr_db','predicted_snr_db','signed_error_db','absolute_error_db']].head(9).to_string(index=False))

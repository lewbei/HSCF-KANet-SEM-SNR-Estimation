import pandas as pd

# SOTA baselines (ARNIQA_SNR, TOPIQ_NR_SNR)
sota = pd.read_csv('sota_iqa_baselines/results/run_metrics.csv')
sota = sota.rename(columns={'model': 'architecture'})

# 11 main architectures
main = pd.read_csv('extracted_dataset/evaluation_results/all_evaluation_results.csv')

common = ['architecture', 'source_dataset', 'test_dataset', 'eval_type', 'seed',
          'mae', 'mse', 'rmse', 'r2', 'n_samples', 'total_params',
          'model_size_mb', 'avg_inference_ms_sample']
sota_c = sota[[c for c in common if c in sota.columns]].copy()
main_c = main[[c for c in common if c in main.columns]].copy()
sota_c['framework'] = 'sota_iqa_baselines'
main_c['framework'] = 'proposed_pipelines'

merged = pd.concat([main_c, sota_c], ignore_index=True)
merged = merged.sort_values(['architecture', 'source_dataset', 'test_dataset', 'seed']).reset_index(drop=True)
out = 'ALL_MODELS_PER_SEED_RESULTS.csv'
merged.to_csv(out, index=False)
print(f"Merged {len(merged)} per-seed evaluation rows -> {out}")
print(f"  architectures: {merged['architecture'].nunique()}")
print(f"  rows per arch: {merged.groupby('architecture').size().to_dict()}")

# Params
total_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions

# GFLOPs
from ptflops import get_model_complexity_info
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=False)
gflops = 2 * macs / 1e9     # multiply-add counted as two FLOPs

# Latency (ms) – median of 100 runs, batch = 1
import time, torch
dummy = torch.randn(1,1,256,256, device='cuda')
model.eval(); model.cuda()
torch.cuda.synchronize()
times = []
for _ in range(120):
    torch.cuda.synchronize()
    t0 = time.time()
    _ = model(dummy)
    torch.cuda.synchronize()
    times.append((time.time()-t0)*1000)
latency_ms = np.median(times[20:])   # skip first 20 warm-ups
print(f"latency = {latency_ms} ")

"""Minimal NCU target: cosmos denoise DiT forward (synthetic real dims, 240p_33f).
Measures achieved SM occupancy / SM throughput of the actual denoise kernels to
answer: does denoise leave SM headroom for concurrent single-GPU multi-staging?"""
import torch
from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs
dev=torch.device("cuda"); dt=torch.bfloat16
model,kw=build_synthetic_inputs("cosmos-predict2.5",batch=1,device=dev,dtype=dt)
B,Cin,_,h,w=kw["hidden_states"].shape
kw["hidden_states"]=torch.randn(B,Cin,33,h,w,device=dev,dtype=dt)  # 33-frame latent
with torch.no_grad():
    for _ in range(2): model(**kw)          # warmup (NCU replays each kernel anyway)
    torch.cuda.synchronize()
    model(**kw)                              # the forward NCU profiles
    torch.cuda.synchronize()
print("done")

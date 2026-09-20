import os
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; refusing root-disk fallback'
"""Rebuild a deterministic full-shape random checkpoint; no released weights."""
import json, time
from pathlib import Path
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler
root=Path('/mnt/nvme/vrl-night-20260918/models/h3_random')
root.mkdir(parents=True,exist_ok=True)
assert not (root/'complete.json').exists()
torch.set_num_threads(4); torch.manual_seed(731)
with init_empty_weights(include_buffers=False):
 model=MiniMaxH3Transformer3DModel()
original_buffers={n:b.clone() for n,b in model.named_buffers()}
start=time.time()
for i,(name,p) in enumerate(list(model.named_parameters())):
 dtype=torch.float32 if any(s in name for s in model._keep_in_fp32_modules) else torch.bfloat16
 if p.ndim>1:
  # CPU BF16 normal_ is scalar/slow; FP32 draws then casting are ~3.6x faster.
  value=torch.empty(p.shape,dtype=torch.float32).normal_(0,0.001).to(dtype)
 elif name.endswith('bias'): value=torch.zeros(p.shape,dtype=dtype)
 else: value=torch.ones(p.shape,dtype=dtype)
 set_module_tensor_to_device(model,name,'cpu',value=value,dtype=dtype)
 if i%100==0: print(i,name,round(time.time()-start,1),flush=True)
for n,b in original_buffers.items(): set_module_tensor_to_device(model,n,'cpu',value=b,dtype=b.dtype)
model.save_pretrained(root/'transformer',max_shard_size='4GB')
MiniMaxH3Scheduler(shift=12).save_pretrained(root/'scheduler')
MiniMaxH3Scheduler(shift=3).save_pretrained(root/'audio_scheduler')
(root/'complete.json').write_text(json.dumps({'seed':731,'generator':'float32_normal_then_native_dtype_v2','parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-start,'scope':'full 33B random base, original computed buffers; no released weights'},indent=2))
print('COMPLETE',flush=True)

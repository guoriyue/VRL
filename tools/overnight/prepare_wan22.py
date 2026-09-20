import os
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; refusing root-disk fallback'
"""Stage pinned Wan2.2 at its runtime storage dtype, preserving FP32 exceptions.

Source shards are streamed individually, avoiding a second full FP32 checkpoint.
No quantization: each tensor has exactly the dtype/value the BF16 loader requests.
"""
import gc, hashlib, json, os, shutil, time
from pathlib import Path
import psutil, torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
repo='Wan-AI/Wan2.2-T2V-A14B-Diffusers'; revision='5be7df9619b54f4e2667b2755bc6a756675b5cd7'
base=Path('/mnt/nvme/vrl-night-20260918/models');root=base/'wan22_bf16';stage=base/'wan22_download_stage';root.mkdir(exist_ok=True);stage.mkdir(exist_ok=True)
keep=('rope','time_embedder','scale_shift_table','norm1','norm2','norm3')
torch.set_num_threads(2); records=[]
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
for item in info.siblings:
 name=item.rfilename
 if not (name.endswith(('.json','.safetensors','.model','.txt')) or name=='model_index.json'): continue
 while psutil.virtual_memory().available<55*2**30 or shutil.disk_usage(base).free<9*2**30:
  print('Waiting for 55 GiB host / 9 GiB tmpfs headroom',flush=True);time.sleep(20)
 dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists(): continue
 digest=getattr(item.lfs,'sha256',None) if item.lfs else None
 cached=next(Path('/home/ubuntu/.cache/huggingface/hub').glob(f'*/blobs/{digest}'),None) if digest else None
 convert=name.startswith(('transformer/','transformer_2/')) and name.endswith('.safetensors')
 if cached and not convert:
  dest.symlink_to(cached);records.append({'file':name,'reused_sha256':digest});continue
 print('Downloading',name,item.size,flush=True)
 source=Path(hf_hub_download(repo,name,revision=revision,local_dir=stage))
 if convert:
  tensors={};preserved=0
  with safe_open(source,framework='pt',device='cpu') as f:
   for key in f.keys():
    tensor=f.get_tensor(key)
    if tensor.is_floating_point() and not any(part in key for part in keep): tensor=tensor.to(torch.bfloat16)
    elif tensor.dtype==torch.float32: preserved+=tensor.numel()
    tensors[key]=tensor
   save_file(tensors,str(dest),metadata=f.metadata())
  records.append({'file':name,'source_sha256':digest,'source_bytes':item.size,'stored_bytes':dest.stat().st_size,'preserved_fp32_numel':preserved})
  del tensors,tensor;gc.collect();source.unlink()
 else:
  source.replace(dest);records.append({'file':name,'source_sha256':digest,'stored_bytes':dest.stat().st_size})
 (root/'preparation_progress.json').write_text(json.dumps(records,indent=2))
 print('Ready',name,flush=True)
(root/'preparation_complete.json').write_text(json.dumps({'repo':repo,'revision':revision,'runtime_dtype':'bfloat16','preserved_fp32_name_patterns':keep,'files':records},indent=2))
print('COMPLETE',flush=True)

"""Matched full Wan 1.3B trainer-only FSDP actor-precision loader control.

Fixed synthetic inputs and nonzero adapters, one backward/AdamW update. This
isolates parent/candidate loading; it does not replace GRPO replay parity.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import torch.distributed as dist
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.loader import load_diffusers_transformer
from vrl.models.families.wan_2_1.model import WanT2VReplayModel
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--primary-fill',action='store_true');p.add_argument('--reference',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(1);torch.manual_seed(731)
rank=int(os.environ['RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(device)
strategy=FSDPStrategy(DistributedTrainingContext(strategy='fsdp',rank=rank,world_size=4,device=device),mesh_dims=['dp_shard'],precision_policy='actor',reshard_after_forward=True,cpu_offload=False,shard_trainable_only=False)
build=ModelBuild(model_name_or_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',revision='0fad780a534b6463e45facd96134c9f345acfa5b',device=torch.device('cpu'),parameter_dtype=torch.bfloat16,family='wan',precision=RolePrecision('bf16','ieee',outer_autocast=False),model_config={'local_files_only':True,'use_lora':True,'lora':{'rank':32,'alpha':64,'parameter_dtype':'float32','target_modules':['to_k','to_q','to_v','to_out.0']}})
kwargs={'materialize_weights':rank==0} if a.primary_fill else {}
model=WanT2VReplayModel(transformer=load_diffusers_transformer(build,'WanTransformer3DModel',**kwargs),scheduler=None,device=device);model.apply_lora(build)
for name,param in model.transformer.named_parameters():
 if param.requires_grad and not param.is_meta:
  generator=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little'))
  with torch.no_grad(): param.copy_(torch.randn(param.shape,generator=generator).mul_(0.001))
model.transformer.enable_gradient_checkpointing();strategy.prepare_model(model)
g=torch.Generator().manual_seed(100+rank)
inputs={'hidden_states':torch.randn(1,16,9,60,104,generator=g).to(device,dtype=torch.bfloat16),'encoder_hidden_states':torch.randn(1,512,4096,generator=g).to(device,dtype=torch.bfloat16),'timestep':torch.tensor([500.],device=device),'return_dict':False}
optimizer=torch.optim.AdamW([p for p in model.transformer.parameters() if p.requires_grad],lr=1e-4)
start=time.perf_counter();pred=model.transformer(**inputs)[0];loss=pred.float().square().mean();strategy.backward(loss)
gradients={};norm=0.
for name,param in model.transformer.named_parameters():
 if param.requires_grad:
  assert param.grad is not None,name
  full=param.grad.full_tensor().detach().cpu();assert torch.isfinite(full).all(),name
  norm+=float(full.double().square().sum())
  if rank==0: gradients[name]=full
assert norm>0
optimizer.step();updated={}
for name,param in model.transformer.named_parameters():
 if param.requires_grad:
  full=param.full_tensor().detach().cpu()
  if rank==0: updated[name]=full
report={'rank':rank,'status':'passed','loss':loss.item(),'grad_norm':norm**.5,'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'primary_fill':a.primary_fill,'scope':'fixed-input one-update loader equivalence; not GRPO'}
if rank==0:
 data={'gradients':gradients,'prediction':pred.detach().cpu(),'updated':updated};torch.save(data,a.output/'tensors.pt')
 if a.reference:
  ref=torch.load(a.reference,map_location='cpu',weights_only=True)
  for field in ['gradients','updated']:
   assert data[field].keys()==ref[field].keys()
   report[field+'_max_abs']=max(float((value-ref[field][name]).abs().max()) for name,value in data[field].items())
  report['prediction_max_abs']=float((data['prediction']-ref['prediction']).abs().max())
  (a.output/'comparison.json').write_text(json.dumps({**report,'status':'observed_not_yet_accepted'},indent=2))
  torch.testing.assert_close(data['prediction'],ref['prediction'],atol=0,rtol=0)
  for field in ['gradients','updated']:
   for name,value in data[field].items(): torch.testing.assert_close(value,ref[field][name],atol=0,rtol=0,msg=name)
  (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2));print(report,flush=True);dist.destroy_process_group()

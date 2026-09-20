"""Matched real SD3.5 FSDP backward, excluding rollout/reward randomness."""
import argparse, hashlib, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.loader import load_diffusers_transformer
from vrl.models.families.sd3_5.model import SD3_5ReplayModel
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--primary-fill',action='store_true');p.add_argument('--reference',type=Path);p.add_argument('--deterministic',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.deterministic:
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
rank=int(os.environ['RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(device);torch.set_num_threads(1);torch.manual_seed(731)
strategy=FSDPStrategy(DistributedTrainingContext(strategy='fsdp',rank=rank,world_size=4,device=device),mesh_dims=['dp_shard'],precision_policy='none',reshard_after_forward=True,cpu_offload=False,shard_trainable_only=True)
build=ModelBuild(model_name_or_path='stabilityai/stable-diffusion-3.5-medium',revision='b940f670f0eda2d07fbb75229e779da1ad11eb80',device=torch.device('cpu'),parameter_dtype=torch.bfloat16,family='sd3_5',precision=RolePrecision('bf16','ieee',outer_autocast=False),model_config={'use_lora':True,'lora':{'rank':32,'alpha':64,'parameter_dtype':'float32','target_modules':['to_k','to_q','to_v','to_out.0','add_k_proj','add_q_proj','add_v_proj','to_add_out']}})
extra={'materialize_weights':rank==0} if a.primary_fill else {}
model=SD3_5ReplayModel(transformer=load_diffusers_transformer(build,'SD3Transformer2DModel',**extra),scheduler=None,device=device);model.apply_lora(build)
for n,v in model.transformer.named_parameters():
 if v.requires_grad and not v.is_meta:
  g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(n.encode()).digest()[:4],'little'))
  with torch.no_grad(): v.copy_(torch.randn(v.shape,generator=g).mul_(0.001))
strategy.prepare_model(model)
g=torch.Generator().manual_seed(100+rank)
inputs={'hidden_states':torch.randn(1,16,64,64,generator=g).to(device,dtype=torch.bfloat16),'encoder_hidden_states':torch.randn(1,128,4096,generator=g).to(device,dtype=torch.bfloat16),'pooled_projections':torch.randn(1,2048,generator=g).to(device,dtype=torch.bfloat16),'timestep':torch.tensor([500.],device=device),'return_dict':False}
start=time.perf_counter();output=model.transformer(**inputs)[0];loss=output.float().square().mean();loss.backward();torch.cuda.synchronize()
gradients={};norm=0.
for n,v in model.transformer.named_parameters():
 if v.requires_grad:
  assert v.grad is not None,n
  full=v.grad.full_tensor().detach().cpu()
  assert torch.isfinite(full).all(),n
  norm+=float(full.double().square().sum())
  if rank==0: gradients[n]=full
report={'rank':rank,'status':'passed','loss':loss.item(),'grad_norm':norm**.5,'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'primary_fill':a.primary_fill}
if rank==0:
 data={'gradients':gradients,'prediction':output.detach().cpu()};torch.save(data,a.output/'tensors.pt')
 if a.reference:
  ref=torch.load(a.reference,map_location='cpu',weights_only=True)
  diff=sum(float((v.double()-ref['gradients'][n].double()).square().sum()) for n,v in gradients.items())
  report['gradient_relative_l2']=(diff/norm)**.5
  report['prediction_max_abs']=float((data['prediction']-ref['prediction']).abs().max())
  # Preserve the artifact even if the strict comparison below fails.
  (a.output/'comparison.json').write_text(json.dumps({**report,'status':'observed_not_yet_accepted'},indent=2))
  torch.testing.assert_close(data['prediction'],ref['prediction'],atol=0,rtol=0)
  assert report['gradient_relative_l2']<=1e-5,report
(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2));print(report,flush=True);dist.destroy_process_group()

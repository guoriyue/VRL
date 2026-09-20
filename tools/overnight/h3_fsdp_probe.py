"""Full-shape trainer-only H3 FSDP versus native block placement.

Random checkpoint, deterministic synthetic replay, FP32 rank-32 adapters,
BF16 frozen base with native FP32 exceptions, one AdamW step. No RL claim.
"""
import argparse, hashlib, json, os, resource, time
from pathlib import Path
import torch
import torch.distributed as dist
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.families.minimax_h3.runtime import build_minimax_h3_replay_runtime_bundle
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['partitioned','fsdp'],required=True)
p.add_argument('--reference',type=Path)
p.add_argument('--no-reshard',action='store_true')
p.add_argument('--small-geometry',action='store_true')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); device=torch.device('cuda',rank if a.mode=='fsdp' else 0)
torch.cuda.set_device(device); torch.set_num_threads(1); torch.manual_seed(731)
devices=[rank] if a.mode=='fsdp' else list(range(4))
for d in devices: torch.cuda.reset_peak_memory_stats(d)
# Keep rank-local capacity evidence even when an experiment fails (e.g. no-reshard).
import sys, traceback
def record_failure(kind, error, tb):
 try:
  report={'status':'failed','mode':a.mode,'rank':rank,'error':repr(error),'reshard_after_forward':not a.no_reshard,'host_process_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_allocated':{str(d):torch.cuda.max_memory_allocated(d) for d in devices},'peak_reserved':{str(d):torch.cuda.max_memory_reserved(d) for d in devices}}
  (a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
 finally: traceback.print_exception(kind,error,tb)
sys.excepthook=record_failure
strategy=None
if a.mode=='fsdp':
 context=DistributedTrainingContext(strategy='fsdp',rank=rank,world_size=world,device=device)
 strategy=FSDPStrategy(context,mesh_dims=['dp_shard'],precision_policy='none',reshard_after_forward=not a.no_reshard,cpu_offload=False,shard_trainable_only=False)
build=ModelBuild(model_name_or_path=a.checkpoint,revision=None,device=device if a.mode=='partitioned' else torch.device('cpu'),parameter_dtype=torch.bfloat16,family='minimax_h3',precision=RolePrecision('bf16','ieee',outer_autocast=False),model_config={'use_lora':True,'lora':{'rank':32,'alpha':64,'parameter_dtype':'float32','target_modules':['to_q','to_k','to_v','to_out.0']},'torch_compile':{'enable':False}},sampling_config={'num_steps':40})
start=time.perf_counter()
import psutil
proc=psutil.Process()
before=proc.memory_info().rss
bundle=build_minimax_h3_replay_runtime_bundle(build,materialize_weights=strategy.materialize_weights if strategy else True,block_devices=tuple(min(i*4//50,3) for i in range(50)) if a.mode=='partitioned' else None)
model=bundle.model; model._device=device
built=proc.memory_info().rss
# Name-derived initialization makes reference and rank-zero adapters identical,
# independent of loader RNG consumption. Both adapter matrices are nonzero.
for n,param in model.transformer.named_parameters():
 if param.requires_grad and not param.is_meta:
  g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(n.encode()).digest()[:4],'little'))
  with torch.no_grad(): param.copy_(torch.randn(param.shape,dtype=param.dtype,generator=g).mul_(0.001).to(param.device))
model.transformer.enable_gradient_checkpointing()
if strategy: strategy.prepare_model(model)
assert not any(x.is_meta for x in model.transformer.parameters())
for d in devices: torch.cuda.synchronize(d)
load_seconds=time.perf_counter()-start
base_allocated={str(d):torch.cuda.memory_allocated(d) for d in devices}
height,width,frames=(128,128,22) if a.small_geometry else (768,1344,124)
latent_t,latent_h,latent_w,audio_t=model._latent_geometry(height,width,frames,vae_geometry=(17,5,16))
g=torch.Generator().manual_seed(912)
latents=torch.randn(1,24,latent_t,latent_h,latent_w,generator=g).to(device)
replay={'num_text_tokens':torch.tensor([512],device=device),'prompt_embeds':torch.randn(1,512,5120,generator=g).to(device),'audio_rows_by_step':torch.randn(1,1,audio_t*2,32,generator=g).to(device)}
context={'num_steps':40,'height':height,'width':width,'num_frames':frames,'vae_geometry':[17,5,16],'fps':24}
state=model.restore_eval_state(replay,context,latents,0)
params=[x for x in model.transformer.parameters() if x.requires_grad]
optimizer=torch.optim.AdamW(params,lr=1e-4)
start=time.perf_counter()
prediction=model.forward_step(state,0)
pred=prediction['noise_pred'];audio_pred=prediction['audio_velocity']
assert torch.isfinite(pred).all() and torch.isfinite(audio_pred).all()
loss=pred.float().square().mean()+audio_pred.float().square().mean()
loss.backward()
for d in devices: torch.cuda.synchronize(d)
compute_seconds=time.perf_counter()-start
reference=torch.load(a.reference,map_location='cpu',weights_only=True) if a.reference else None
comparison={}; saved={}; nonzero=0; sqnorm=0.0; error_sq=0.0; reference_sq=0.0
if reference:
 torch.testing.assert_close(pred.detach().cpu(),reference['prediction'],atol=1e-3,rtol=1e-3)
 comparison['prediction_max_abs']=float((pred.detach().cpu()-reference['prediction']).abs().max())
 torch.testing.assert_close(audio_pred.detach().cpu(),reference['audio_prediction'],atol=1e-3,rtol=1e-3)
 comparison['audio_prediction_max_abs']=float((audio_pred.detach().cpu()-reference['audio_prediction']).abs().max())
for n,param in model.transformer.named_parameters():
 if not param.requires_grad: continue
 assert param.grad is not None,n
 grad=param.grad.full_tensor() if hasattr(param.grad,'full_tensor') else param.grad
 grad=grad.detach().cpu()
 assert torch.isfinite(grad).all(),n
 nonzero+=bool(torch.count_nonzero(grad)); sqnorm+=float(grad.double().square().sum())
 if rank==0:
  saved[n]=grad
  if reference:
   expected=reference['gradients'][n]
   error_sq+=float((grad.double()-expected.double()).square().sum())
   reference_sq+=float(expected.double().square().sum())
   torch.testing.assert_close(grad,expected,atol=1e-3,rtol=1e-3,msg=n)
   comparison.setdefault('gradient_max_abs',0.0)
   comparison['gradient_max_abs']=max(comparison['gradient_max_abs'],float((grad-expected).abs().max()))
assert nonzero>0
if reference and rank==0:
 comparison['gradient_relative_l2']=(error_sq/max(reference_sq,1e-30))**0.5
 assert comparison['gradient_relative_l2']<=0.01,comparison
first=params[0]; local=first.to_local() if hasattr(first,'to_local') else first
before_step=local.detach().clone()
optimizer.step()
assert not torch.equal(local,before_step),'Optimizer did not change first adapter'
for d in devices: torch.cuda.synchronize(d)
report={'mode':a.mode,'rank':rank,'world_size':world,'status':'passed','loss':loss.item(),'gradient_norm':sqnorm**0.5,'nonzero_gradient_tensors':nonzero,'geometry':[height,width,frames],'load_seconds':load_seconds,'forward_backward_seconds':compute_seconds,'rss_before_build':before,'rss_after_build':built,'host_process_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'resident_after_prepare':base_allocated,'peak_allocated':{str(d):torch.cuda.max_memory_allocated(d) for d in devices},'peak_reserved':{str(d):torch.cuda.max_memory_reserved(d) for d in devices},'comparison':comparison,'reshard_after_forward':not a.no_reshard}
(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
if rank==0 and a.mode=='partitioned': torch.save({'prediction':pred.detach().cpu(),'audio_prediction':audio_pred.detach().cpu(),'gradients':saved},a.output/'reference.pt')
print(json.dumps(report),flush=True)
if dist.is_initialized(): dist.destroy_process_group()

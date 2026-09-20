"""Real dual-expert, full-geometry parity and backward admission for Wan14B."""
import argparse,json,os,gc,time
from pathlib import Path
import torch
import torch.distributed as dist
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.loader import load_diffusers_transformer,load_diffusers_scheduler
from vrl.models.families.wan_2_1.model import WanT2VReplayModel
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK']);dev=torch.device('cuda',rank);torch.cuda.set_device(dev);torch.set_num_threads(1);torch.manual_seed(20260914)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
build=ModelBuild(model_name_or_path='/mnt/nvme/vrl-night-20260918/models/wan22_bf16',revision=None,device=torch.device('cpu'),parameter_dtype=torch.bfloat16,family='wan',precision=RolePrecision('bf16','ieee',outer_autocast=True),model_config={'local_files_only':True,'use_lora':True,'trainable_transformers':'both','lora':{'rank':32,'alpha':64,'parameter_dtype':'float32','target_modules':['add_k_proj','add_q_proj','add_v_proj','to_add_out','to_k','to_out.0','to_q','to_v']}})
scheduler=load_diffusers_scheduler(build,'UniPCMultistepScheduler');scheduler.set_timesteps(20,device=dev)
model=WanT2VReplayModel(transformer=load_diffusers_transformer(build,'WanTransformer3DModel'),transformer_2=load_diffusers_transformer(build,'WanTransformer3DModel',subfolder='transformer_2'),scheduler=scheduler,device=dev,boundary_ratio=.875,trainable_transformers='both')
for m in model.trainable_modules.values():m.requires_grad_(False)
model.apply_lora(build);model.precision=build.precision
context={'guidance_scale':4.5,'cfg':True,'boundary_ratio':.875,'num_train_timesteps':1000}
g=torch.Generator().manual_seed(42+rank);obs=torch.randn(1,16,9,60,104,generator=g).to(dev);pe=torch.randn(1,256,4096,generator=g).to(dev);ne=torch.randn(1,256,4096,generator=g).to(dev)
cases=[]
for name,idx in [('transformer',0),('transformer_2',12)]:
 module=getattr(model,name);module.to(dev);module.eval();ts=scheduler.timesteps[idx:idx+1]
 state=model.restore_eval_state({'timesteps':ts[:,None],'prompt_embeds':pe,'negative_prompt_embeds':ne},context,obs,0)
 with torch.no_grad():
  pred=model.forward_step(state,0)['noise_pred']
  transition=sde_step_with_logprob(scheduler,pred,ts,obs,step_index=idx,generator=torch.Generator(device=dev).manual_seed(919+rank))
 cases.append((name,idx,state,pred.detach(),transition))
 module.to('cpu');gc.collect();torch.cuda.empty_cache()
 print(json.dumps({'rank':rank,'event':'unsharded_reference','expert':name}),flush=True)
for m in model.trainable_modules.values():m.train();m.enable_gradient_checkpointing()
strategy=FSDPStrategy(DistributedTrainingContext(strategy='fsdp',rank=rank,world_size=4,device=dev),mesh_dims=['dp_shard'],precision_policy='none',reshard_after_forward=True,cpu_offload=False,shard_trainable_only=False)
strategy.prepare_model(model);params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-4)
torch.cuda.reset_peak_memory_stats();results=[]
for name,idx,state,oldpred,transition in cases:
 pred=model.forward_step(state,0)['noise_pred'];result=sde_step_with_logprob(scheduler,pred,state.timesteps[0],obs,prev_sample=transition.prev_sample,step_index=idx)
 report={'expert':name,'prediction_max_abs':float((pred.detach()-oldpred).abs().max()),'parity_max_abs':float((result.log_prob.detach()-transition.log_prob).abs().max())}
 (a.output/f'observed-{rank}-{name}.json').write_text(json.dumps(report,indent=2));print(json.dumps({'rank':rank,**report}),flush=True)
 assert report['parity_max_abs']==0,report
 strategy.backward(-result.log_prob.mean());results.append(report)
 del pred,result
norm=torch.zeros((),device=dev,dtype=torch.float64)
for p in params:
 assert p.grad is not None
 g=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
 assert torch.isfinite(g).all();norm+=g.double().square().sum()
dist.all_reduce(norm);assert norm>0;optimizer.step()
report={'status':'passed','rank':rank,'cases':results,'grad_norm':float(norm.sqrt()),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'scope':'Real dual-expert weights, CFG4.5, 480x832x33, batch1, full FSDP; diagnostic admission, not online acceptance'}
(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);dist.destroy_process_group()

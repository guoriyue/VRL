"""Full-geometry Wan SDE roundtrip under matched unsharded/FSDP execution.
Diagnostic only: fresh transitions, batch 1, eager, actor BF16 on both sides,
lossless trajectory. Does not substitute for the original online recipe gate.
"""
import os,glob,json,time,argparse
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
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK']);dev=torch.device('cuda',rank);torch.cuda.set_device(dev);torch.set_num_threads(1);torch.manual_seed(731)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
x=torch.load(sorted(glob.glob('/mnt/nvme/vrl-night-20260918/fixtures/wan13_spool/.*/0.pt'))[0],map_location='cpu',weights_only=False).batches[0]
t=x.trajectory.segments['denoise'].tensors;step=int(t['sde_window'].value[rank,0]);ctx=x.context
obs=t['observations'].value[rank:rank+1,step].float().to(dev);ts=t['timesteps'].value[rank:rank+1,step].to(dev);pe=t['prompt_embeds'].value[rank:rank+1].to(dev);ne=t['negative_prompt_embeds'].value[rank:rank+1].to(dev);del t,x
build=ModelBuild(model_name_or_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',revision='0fad780a534b6463e45facd96134c9f345acfa5b',device=torch.device('cpu'),parameter_dtype=torch.bfloat16,family='wan',precision=RolePrecision('bf16','tf32',outer_autocast=True),model_config={'local_files_only':True,'use_lora':True,'lora':{'rank':32,'alpha':64,'target_modules':['to_k','to_q','to_v','to_out.0']}})
scheduler=load_diffusers_scheduler(build,'UniPCMultistepScheduler');scheduler.set_timesteps(20,device=dev)
model=WanT2VReplayModel(transformer=load_diffusers_transformer(build,'WanTransformer3DModel'),scheduler=scheduler,device=dev);model.apply_lora(build);model.precision=build.precision;model.transformer.to(device=dev,dtype=torch.bfloat16)
state=model.restore_eval_state({'timesteps':ts[:,None],'prompt_embeds':pe,'negative_prompt_embeds':ne},ctx,obs,0)
start=time.time()
with torch.no_grad():
 model.transformer.eval();oldpred=model.forward_step(state,0)['noise_pred']
 transition=sde_step_with_logprob(scheduler,oldpred,ts,obs,step_index=step,generator=torch.Generator(device=dev).manual_seed(42+rank))
print(json.dumps({'rank':rank,'event':'unsharded_rollout_done','old_logprob':transition.log_prob.tolist()}),flush=True)
model.transformer.train();model.transformer.enable_gradient_checkpointing()
strategy=FSDPStrategy(DistributedTrainingContext(strategy='fsdp',rank=rank,world_size=4,device=dev),mesh_dims=['dp_shard'],precision_policy='actor',reshard_after_forward=True,cpu_offload=False,shard_trainable_only=False)
strategy.prepare_model(model)
optimizer=torch.optim.AdamW([p for p in model.transformer.parameters() if p.requires_grad],lr=1e-4)
newpred=model.forward_step(state,0)['noise_pred'];result=sde_step_with_logprob(scheduler,newpred,ts,obs,prev_sample=transition.prev_sample,step_index=step)
report={'rank':rank,'scope':'aligned execution diagnostic, not original recipe acceptance','prediction_max_abs':float((newpred.detach().float()-oldpred.float()).abs().max()),'parity_max_abs':float((result.log_prob.detach()-transition.log_prob).abs().max()),'old_logprob':transition.log_prob.tolist(),'new_logprob':result.log_prob.detach().tolist()}
(a.output/f'observed{rank}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
assert report['parity_max_abs']==0,report
strategy.backward(-result.log_prob.mean());normsq=torch.zeros((),device=dev,dtype=torch.float64)
for p in model.transformer.parameters():
 if p.requires_grad:
  assert p.grad is not None
  g=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
  assert torch.isfinite(g).all();normsq+=g.double().square().sum()
dist.all_reduce(normsq);report['grad_norm']=float(normsq.sqrt());assert report['grad_norm']>0
optimizer.step();report.update(status='passed',seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated());(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);dist.destroy_process_group()

"""Four independent GPU diagnostics; never grants an H3 acceptance gate.
Uses a saved original-recipe group, plus fresh one-step SDE draws to isolate
storage loss from backbone differences. No rewards or video regeneration.
"""
import os, json, glob, argparse, time
from pathlib import Path
import torch
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.loader import load_diffusers_transformer, load_diffusers_scheduler
from vrl.models.families.wan_2_1.model import WanT2VReplayModel
from vrl.math.denoise.flow_matching import sde_step_with_logprob
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ.get('LOCAL_RANK',0));device=torch.device('cuda',rank);torch.cuda.set_device(device);torch.set_num_threads(1);torch.manual_seed(731)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
actor=rank in (1,3);compiled=rank>=2
report={'rank':rank,'actor_cast':actor,'compiled':compiled,'scope':'diagnostic, not acceptance'}
def emit(event,**kw):print(json.dumps({'rank':rank,'event':event,**kw}),flush=True)
x=torch.load(sorted(glob.glob('/mnt/nvme/vrl-night-20260918/fixtures/wan13_spool/.*/0.pt'))[0],map_location='cpu',weights_only=False).batches[0]
t=x.trajectory.segments['denoise'].tensors;step=int(t['sde_window'].value[0,0]);ctx=x.context
obs=t['observations'].value[:,step].to(device);action=t['actions'].value[:,step].to(device);old=t['old_log_prob'].value[:,step].to(device).float();ts=t['timesteps'].value[:,step].to(device)
pe=t['prompt_embeds'].value.to(device);ne=t['negative_prompt_embeds'].value.to(device);del t,x
build=ModelBuild(model_name_or_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',revision='0fad780a534b6463e45facd96134c9f345acfa5b',device=torch.device('cpu'),parameter_dtype=torch.bfloat16,family='wan',precision=RolePrecision('bf16','tf32',outer_autocast=True),model_config={'local_files_only':True,'use_lora':True,'lora':{'rank':32,'alpha':64,'target_modules':['to_k','to_q','to_v','to_out.0']}})
scheduler=load_diffusers_scheduler(build,'UniPCMultistepScheduler');scheduler.set_timesteps(20,device=device)
model=WanT2VReplayModel(transformer=load_diffusers_transformer(build,'WanTransformer3DModel'),scheduler=scheduler,device=device);model.apply_lora(build);model.precision=build.precision
report['native_fp32_parameters']=[n for n,p in model.transformer.named_parameters() if p.dtype==torch.float32]
if actor:model.transformer.to(dtype=torch.bfloat16)
model.transformer.to(device);model.transformer.eval()
if compiled:model.transformer=torch.compile(model.transformer,mode='default')
def forward(o,p,n,t):
 state=model.restore_eval_state({'timesteps':t[:,None],'prompt_embeds':p,'negative_prompt_embeds':n},ctx,o,0)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model.forward_step(state,0)['noise_pred']
def score(noise,o,act):return sde_step_with_logprob(scheduler,noise,ts,o,prev_sample=act,step_index=step).log_prob
start=time.time();emit('loaded',step=step)
pred=forward(obs,pe,ne,ts);emit('batch4_forward_done',seconds=time.time()-start)
logp=score(pred,obs,action)
report['saved_old_logprob']=old.tolist();report['saved_replay_logprob']=logp.tolist();report['saved_max_abs_diff']=float((logp-old).abs().max())
# Exactly the same prediction and sampled transition: storage alone can fail.
g=torch.Generator(device=device).manual_seed(42)
fresh=sde_step_with_logprob(scheduler,pred,ts,obs,step_index=step,generator=g)
report['same_inputs_exact_replay_diff']=float((score(pred,obs,fresh.prev_sample)-fresh.log_prob).abs().max())
report['logprob_storage_only_diff']=float((fresh.log_prob.to(torch.bfloat16).float()-fresh.log_prob).abs().max())
report['action_storage_only_diff']=float((score(pred,obs,fresh.prev_sample.to(torch.bfloat16))-fresh.log_prob).abs().max())
report['combined_storage_diff']=float((score(pred,obs,fresh.prev_sample.to(torch.bfloat16))-fresh.log_prob.to(torch.bfloat16).float()).abs().max())
emit('storage_measurements',**{k:v for k,v in report.items() if k.endswith('diff')})
# Original replay is sample-wise: isolate GEMM/attention batch-shape drift.
single=torch.cat([forward(obs[i:i+1],pe[i:i+1],ne[i:i+1],ts[i:i+1]) for i in range(4)])
report['batch4_vs_batch1_prediction_max']=float((pred.float()-single.float()).abs().max())
report['batch4_vs_batch1_logprob_max']=float((score(single,obs,action)-logp).abs().max())
torch.save({'prediction':pred.cpu(),'single_prediction':single.cpu(),'logprob':logp.cpu()},a.output/f'tensors{rank}.pt')
report.update(status='passed',seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated())
(a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2));emit('finished',**{k:v for k,v in report.items() if k not in ('rank','native_fp32_parameters')})

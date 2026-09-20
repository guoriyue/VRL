"""Three requested 20-update arms. Drift is diagnostic, not a zero-parity claim."""
import csv,json,math,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918/miles';out.mkdir(parents=True,exist_ok=True)
arms={
 'A_no_compile':['model.torch_compile.enable=false'],
 'B_no_batch16':['rollout.samples_per_generation_batch=1'],
 'C_no_recompute':['trainer.precision_correction.recompute_old_logprob=off','trainer.debug.max_abs_logprob_diff=0.05'],
}
reports={}
for name,overrides in arms.items():
 dest=out/name
 cmd=[sys.executable,'-m','vrl.scripts.train','--config','experiment/sd3_5/online_grpo_ocr_dedicated_3x1_recompute','trainer.total_epochs=20',f'trainer.output_dir={dest}',*overrides]
 (out/f'{name}.command.json').write_text(json.dumps(cmd,indent=2))
 with (out/f'{name}.log').open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 try:
  assert result.returncode==0,f'exit {result.returncode}'
  verdict=json.loads((dest/'training_run_result.rank-0.json').read_text());assert verdict['status']=='success',verdict
  rows=list(csv.DictReader((dest/'metrics.full_precision.csv').open()));assert len(rows)==20,len(rows)
  for row in rows:
   for key in ['loss','reward_mean','grad_norm','pre_update_logprob_abs_diff_max']: assert math.isfinite(float(row[key])),row
   assert float(row['grad_norm'])>0,row
  reports[name]={'status':'passed','updates':len(rows),'first_reward':float(rows[0]['reward_mean']),'last_reward':float(rows[-1]['reward_mean']),'max_rollout_replay_drift':max(float(r['pre_update_logprob_abs_diff_max']) for r in rows),'scope':'ablation completed; not strict replay parity acceptance'}
 except Exception as error: reports[name]={'status':'failed','error':str(error)}
 (out/'arms.json').write_text(json.dumps(reports,indent=2))
assert all(r['status']=='passed' for r in reports.values()),reports
(out/'rank0.json').write_text(json.dumps({'status':'passed','arms':reports},indent=2))

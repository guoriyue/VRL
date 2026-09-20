"""Post-run runtime-health evidence, separate from numerical parity."""
import json,re,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918';dest=out/'wan13_clean_health';dest.mkdir(exist_ok=True)
log=(out/'primary_fill_wan13_clean.launch.log').read_text(errors='replace')
failures=[line for line in log.splitlines() if re.search(r'Workers .*killed due to memory pressure|OutOfMemoryError|CUDA out of memory|NVRM: Xid',line)]
report={'runtime_failures':failures,'scope':'Runtime health only; does not change zero-parity numerical gate'}
source=Path('/dev/shm/vrl-night-20260918/runs/wan13_clean');checkpoint=source/'checkpoint-final';target=out/'receipts/wan13_clean/checkpoint-final'
if (checkpoint/'checkpoint_meta.json').exists() and not target.exists():
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(checkpoint,target)
report['final_checkpoint_persisted']=target.exists()
for rank in range(4):
 try:
  verdict=json.loads((source/f'training_run_result.rank-{rank}.json').read_text())
  if verdict['status']!='success':failures.append(f'rank{rank}: {verdict}')
 except Exception as e:failures.append(f'rank{rank}: {e}')
report['status']='failed' if failures else 'passed'
trace=out/'primary_fill_wan13_clean.resources.jsonl'
if trace.exists():
 rows=[json.loads(line) for line in trace.read_text().splitlines()]
 if rows:
  report['minimum_host_available_gib']=min(r['host_available'] for r in rows)/2**30
  report['maximum_process_pss_sum_gib']=max(sum(p.get('pss') or 0 for p in r['processes']) for r in rows)/2**30
(dest/'rank0.json').write_text(json.dumps(report,indent=2));assert not failures,failures

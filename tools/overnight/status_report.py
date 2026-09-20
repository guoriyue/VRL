"""Refresh a compact on-disk status report without touching queue ownership."""
import csv,json,time
from datetime import datetime
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918'
while True:
 try:
  q=json.loads((out/'queue.json').read_text());by={j['id']:j for j in q['jobs']}
  lines=['# Four-L40S overnight status','',f'Updated {datetime.now().astimezone().isoformat(timespec="seconds")}','',f'Production commit: `{q["commit"]}`. Failed correctness gates block dependents.','', '| Job | State | Waiting for |','| --- | --- | --- |']
  for j in q['jobs']:
   dependencies=[d for d in j.get('depends_on',[]) if by.get(d,{}).get('status')!='passed']
   lines.append(f'| {j["id"]} | {j["status"]} | {", ".join(dependencies)} |')
  lines+=['','## Observed online updates','']
  for name in ['sd35','sd35_parent','wan13_retry']:
   path=out/name/'metrics.full_precision.csv'
   rows=list(csv.DictReader(path.open())) if path.exists() else []
   if rows:
    values=[{k:r.get(k) for k in ['epoch','loss','grad_norm','reward_mean','pre_update_logprob_abs_diff_max']} for r in rows]
    lines+=[f'### {name}','', '```json',json.dumps(values,indent=2),'```','']
  lines+=['H3 synthetic trainer-only and fixed-input loader probes do not establish released-weight quality or GRPO learning. The Wan 14B fallback is conditional and must not be described as started until its job is running.','']
  tmp=out/'STATUS.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(out/'STATUS.md')
 except Exception as e:print(repr(e),flush=True)
 time.sleep(30)

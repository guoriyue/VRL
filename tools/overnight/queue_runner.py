"""Exclusive, appendable experiment queue with explicit success gates."""
import csv, fcntl, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[2]; out=root/'outputs/overnight_20260918'
lock=(out/'runner.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def alive(pid):
 try: return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess: return False
def update(jid,**fields):
 with (out/'queue.edit.lock').open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  q=json.loads((out/'queue.json').read_text())
  job=next(j for j in q['jobs'] if j['id']==jid); job.update(fields)
  temp=out/'queue.json.tmp'; temp.write_text(json.dumps(q,indent=2)); temp.replace(out/'queue.json')
def gate(job):
 path=Path(job['output'])
 if job.get('gate')=='training':
  verdicts=[json.loads((path/f'training_run_result.rank-{r}.json').read_text()) for r in range(4)]
  assert all(v['status']=='success' for v in verdicts),verdicts
  rows=list(csv.DictReader((path/'metrics.full_precision.csv').open()))
  assert len(rows)>=job['updates'],f'Only {len(rows)} updates'
  for row in rows:
   for k in ('loss','grad_norm','reward_mean'): assert math.isfinite(float(row[k])),(k,row[k])
   assert float(row['grad_norm'])>0
   assert float(row['pre_update_logprob_abs_diff_max'])<=job['parity_limit'],row
  if job.get('compare_first_update'):
   expected=list(csv.DictReader((Path(job['compare_first_update'])/'metrics.full_precision.csv').open()))[0]
   comparison={k:{'actual':float(rows[0][k]),'expected':float(expected[k])} for k in ('loss','grad_norm','reward_mean','pre_update_logprob_abs_diff_max')}
   (path/'parent_comparison.json').write_text(json.dumps(comparison,indent=2))
   for k,values in comparison.items():
    assert math.isclose(values['actual'],values['expected'],rel_tol=1e-5,abs_tol=1e-8),(k,values)
  (path/'queue_gate.json').write_text(json.dumps({'status':'passed','rows':rows,'parity_limit':job['parity_limit']},indent=2))
 elif job.get('gate')=='probe':
  verdicts=[json.loads((path/f'rank{r}.json').read_text()) for r in range(job.get('ranks',4))]
  assert all(v['status']=='passed' for v in verdicts)
 else: raise ValueError('Missing acceptance gate')
def finish(job,returncode=None):
 try:
  if returncode not in (None,0): raise RuntimeError(f'exit {returncode}')
  gate(job); update(job['id'],status='passed',finished=time.time(),returncode=returncode)
 except Exception as e: update(job['id'],status='failed',finished=time.time(),error=str(e),returncode=returncode)
deadline=time.time()+float(os.environ.get('VRL_QUEUE_HOURS','12'))*3600
while time.time()<deadline:
 q=json.loads((out/'queue.json').read_text()); jobs=q['jobs']; states={j['id']:j['status'] for j in jobs}
 adopted=next((j for j in jobs if j['status']=='running'),None)
 if adopted:
  if not alive(adopted['pid']): finish(adopted)
  time.sleep(5); continue
 job=next((j for j in jobs if j['status']=='ready' and all(states.get(d)=='passed' for d in j.get('depends_on',[]))),None)
 if job is None: time.sleep(10); continue
 if any(not Path(p).exists() for p in job.get('requires',[])): time.sleep(10); continue
 inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if inventory:
  print('Waiting for GPU processes',inventory,flush=True); time.sleep(10); continue
 env=os.environ.copy(); env.update(job.get('env',{})); env.update(PYTHONPATH=job.get('pythonpath_prefix','')+job.get('cwd',str(root)),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 with (out/(job['id']+'.launch.log')).open('w') as log:
  proc=subprocess.Popen(job['command'],cwd=job.get("cwd",root),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 update(job['id'],status='running',pid=proc.pid,started=time.time())
 monitor=subprocess.Popen(['/home/ubuntu/VRL/.venv/bin/python',str(root/'tools/overnight/monitor.py'),str(proc.pid),str(out/(job['id']+'.resources.jsonl'))],cwd=root)
 try: rc=proc.wait(timeout=min(job.get('timeout_seconds',21600),max(1,deadline-time.time())))
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try: proc.wait(timeout=45)
  except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
  rc=124
 finish(job,rc)
print('Queue monitoring window ended',flush=True)

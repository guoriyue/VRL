"""Append a reviewed job JSON before the fallback long run; serialize edits."""
import argparse,fcntl,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('job',type=Path);a=p.parse_args()
out=Path(__file__).resolve().parents[2]/'outputs/overnight_20260918'
job=json.loads(a.job.read_text())
assert all(k in job for k in ['id','command','gate','output'])
assert isinstance(job['command'],list) and job['command']
assert job['gate'] in ['training','probe']
job.setdefault('status','ready')
with (out/'queue.edit.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads((out/'queue.json').read_text());jobs=q['jobs']
 assert not any(j['id']==job['id'] for j in jobs),'Duplicate job id'
 long=next(j for j in jobs if j['id']=='wan14b_long_run')
 assert long['status']!='running','Long run already active; enqueue requires deciding checkpoint boundary first'
 index=jobs.index(long);jobs.insert(index,job);long.setdefault('depends_on',[]).append(job['id'])
 temp=out/'queue.json.edit';temp.write_text(json.dumps(q,indent=2));temp.replace(out/'queue.json')
print('Queued',job['id'])

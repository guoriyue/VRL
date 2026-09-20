"""Read-only replay progress and timing from completed model-forward trace events."""
import argparse,datetime,json,re,statistics
from pathlib import Path
from zoneinfo import ZoneInfo
import psutil
p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=json.loads(a.status.read_text());directory=a.status.resolve().parent
log=directory/('training.log' if directory.name=='continuous' else 'smoke.log')
rows={}
for line in log.read_text(errors='replace').splitlines():
 if 'WAN_EXPERT_LIFECYCLE {' not in line or 'RayGenerationWorker' in line:continue
 rank=re.search(r'\[default(\d)\]',line);ts=re.search(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+',line)
 if not rank or not ts:continue
 e=json.loads(line.split('WAN_EXPERT_LIFECYCLE ',1)[1]);e['time_unix_s']=datetime.datetime.strptime(ts[0],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
 rows.setdefault(rank[1],[]).append(e)
report={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status_file':str(a.status.resolve()),'training_pid':s.get('training_pid'),'training_process_alive':psutil.pid_exists(s.get('training_pid',0)),'scope':'Completed replay forward trace events; NOT completed optimizer updates or parity acceptance. Inter-event time includes previous backward plus next forward and orchestration.','ranks':{}}
for rank,events in rows.items():
 dt=[b['time_unix_s']-a['time_unix_s'] for a,b in zip(events,events[1:])]
 report['ranks'][rank]={'completed_forward_calls':len(events),'last_expert':events[-1]['expert'],'last_timestep':events[-1]['timestep'],'first_time_unix_s':events[0]['time_unix_s'],'last_time_unix_s':events[-1]['time_unix_s'],'median_completion_interval_seconds':statistics.median(dt) if dt else None,'max_forward_peak_allocated_bytes':max(e['peak_allocated_bytes'] for e in events),'last_reserved_bytes':events[-1]['reserved_after_bytes']}
a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output);print(json.dumps(report,indent=2))

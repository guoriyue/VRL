"""Immutable evidence snapshot for comparing a future Wan staging implementation."""
import argparse,csv,datetime,hashlib,json,re,shutil,statistics,subprocess,sys
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=a.out/stamp;out.mkdir(parents=True)
def save(name,value):
 (out/name).write_text(json.dumps(value,indent=2))
def copy(src,dst):
 if src.exists():
  target=out/dst;target.parent.mkdir(parents=True,exist_ok=True)
  # Snapshot the bytes available now; concurrent appends do not alter this copy.
  target.write_bytes(src.read_bytes())
def timestamp(line):
 m=re.search(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+',line)
 return datetime.datetime.strptime(m[0],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('America/Los_Angeles')).timestamp() if m else None
summary=[];gpu_rows=[];batch_rows=[]
base=ROOT/'outputs/wan14b_priority_20260919'
for attempt in (3,4,5):
 name=f'attempt{attempt:02d}';src=base/name;dest=out/name;dest.mkdir()
 for fn in ('base.yaml','smoke.yaml','smoke.command.json','status.json','smoke.acceptance.json','smoke.log','smoke.resources.jsonl'):
  copy(src/fn,Path(name)/fn)
 status=json.loads((dest/'status.json').read_text());run=Path(status['output'])
 for fn in ('resolved_config.yaml','metrics.full_precision.csv','training_debug.jsonl','reward_debug/kling_video_reward_results.jsonl'):
  copy(run/fn,Path(name)/fn)
 for f in (run/'run_evidence').glob('*.json'):copy(f,Path(name)/'run_evidence'/f.name)
 for f in run.glob('training_run_result.rank-*.json'):copy(f,Path(name)/f.name)
 walls={};first={};end={};errors=[]
 for line in (dest/'smoke.log').read_text(errors='replace').splitlines():
  r=re.search(r'\[default(\d)\]',line)
  if not r:continue
  rank=int(r[1]);t=timestamp(line)
  if 'WAN_EXPERT_LIFECYCLE {' in line and t:first.setdefault(rank,t)
  m=re.search(r'generation wall:.*wall_s=([\d.]+)',line)
  if m:walls[rank]=float(m[1]);end[rank]=t
  m=re.search(r'batch=(\S+).*samples=(\d+).*queue_wait=([\d.]+)s exec=([\d.]+)s',line)
  if m:batch_rows.append({'attempt':name,'rank':rank,'batch':m[1],'samples':int(m[2]),'queue_wait_s':float(m[3]),'execution_s':float(m[4])})
  if 'ERROR online recipe failed' in line:errors.append(re.sub(r'\x1b\[[0-9;]*m','',line))
 resources=[]
 for line in (dest/'smoke.resources.jsonl').read_text().splitlines():
  try:resources.append(json.loads(line))
  except json.JSONDecodeError:pass
 per_gpu={}
 for row in resources:
  for g in row['gpu'].splitlines():
   vals=g.split(',');
   if len(vals)!=4:continue
   idx=int(vals[0]);entry={'attempt':name,'time_unix_s':row['time'],'gpu':idx,'memory_mib':float(vals[1]),'utilization_pct':float(vals[2]),'power_w':float(vals[3]),'host_available_gib':row['host_available']/2**30}
   entry['generation_compute_window']=idx in first and idx in end and first[idx]<=row['time']<=end[idx]
   gpu_rows.append(entry);per_gpu.setdefault(idx,[]).append(entry)
 gpu_stats={}
 for idx,rows in per_gpu.items():
  covered=energy=util=0.
  for x,y in zip(rows,rows[1:]):
   if idx not in first or idx not in end:continue
   dt=max(0,min(y['time_unix_s'],end[idx])-max(x['time_unix_s'],first[idx]))
   covered+=dt;energy+=x['power_w']*dt;util+=x['utilization_pct']*dt
  gpu_stats[idx]={'peak_memory_mib_sampled':max(x['memory_mib'] for x in rows),'generation_window_covered_s':covered,'generation_utilization_pct_time_weighted':util/covered if covered else None,'generation_gpu_energy_kwh_estimate':energy/3600000 if covered else None}
 rewards=[];rp=dest/'reward_debug/kling_video_reward_results.jsonl'
 if rp.exists():
  for line in rp.read_text().splitlines():
   try:rewards.append(json.loads(line))
   except json.JSONDecodeError:pass
 reward_stats={k:{'n':len(rewards),'mean':statistics.mean(r['scores'][k] for r in rewards),'std_population':statistics.pstdev(r['scores'][k] for r in rewards)} for k in rewards[0]['scores']} if rewards else {}
 metrics=list(csv.DictReader((dest/'metrics.full_precision.csv').open())) if (dest/'metrics.full_precision.csv').exists() else []
 duration=max(walls.values()) if len(walls)==4 else None
 summary.append({'attempt':name,'status_at_snapshot':status['status'],'completed_updates_in_metrics':len(metrics),'generation_wall_s_by_rank':walls,'generation_32_videos_wall_s_max_rank':duration,'aggregate_generation_videos_per_hour':32*3600/duration if duration else None,'minimum_host_available_gib_sampled':min(r['host_available'] for r in resources)/2**30,'gpu':gpu_stats,'reward':reward_stats,'first_errors':errors[:4],'online_acceptance':(dest/'smoke.acceptance.json').exists()})
save('summary.json',{'captured_at_utc':stamp,'attempts':summary,'notes':['Log timestamps are America/Los_Angeles; converted to Unix UTC for alignment with resource samples.','Attempt 5 is a live snapshot; missing update metrics are not zeros.','Generation wall is executor-reported time, slowest of four ranks, 32 videos total.','GPU utilization/energy integrates sampled NVML values over first denoise event to generation completion per rank; excludes model startup. Energy is GPU only, not system energy.','Memory peaks are sampled and can miss short-lived allocations. Host available is the OS available metric, not a sum of process RSS.','This is not a staging A/B: attempts also differ in microbatch and memory handling. Compare new staging against attempt 5 with its exact settings.','No model-quality improvement claim.']})
for filename,rows in [('gpu_timeseries.csv',gpu_rows),('batch_timings.csv',batch_rows)]:
 with (out/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for f in (base/'replay_preflight_batch1_v2').glob('*.json'):copy(f,Path('replay_preflight')/f.name)
for f in (ROOT/'tools/overnight').glob('wan14b*.py'):copy(f,Path('source/tools')/f.name)
for fn in ('monitor.py','queue_runner.py','export_staging_baseline.py'):copy(ROOT/'tools/overnight'/fn,Path('source/tools')/fn)
(out/'source/tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT))
save('source/provenance.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'worktree':str(ROOT),'source_capture_time_utc':stamp,'caveat':'Current worktree diff at snapshot time; earlier attempts used earlier local revisions. Original attempt configs and logs are archived separately.'})
(out/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv'],text=True))
import importlib.metadata
save('versions.json',{k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','accelerate','peft','ray']})
for model in ('wan22_bf16','KlingVideoReward'):
 copy(Path('/mnt/nvme/vrl-night-20260918/models')/model/'preparation_complete.json',Path('models')/(model+'.json'))
lines=['# Wan 14B staging comparison baseline','',f'Snapshot: {stamp}. Current attempt is provisional; no completed online update at capture unless shown below.','','| Attempt | Batch | Generation wall (32 videos) | Videos/hour | Min host available | Updates |','|---|---:|---:|---:|---:|---:|']
for s in summary:
 wall=s['generation_32_videos_wall_s_max_rank'];rate=s['aggregate_generation_videos_per_hour'];batch=1 if s['attempt']=='attempt05' else 2
 lines.append(f"| {s['attempt']} | {batch} | {wall:.1f} s | {rate:.2f} | {s['minimum_host_available_gib_sampled']:.1f} GiB | {s['completed_updates_in_metrics']} |" if wall else f"| {s['attempt']} | {batch} | incomplete | — | {s['minimum_host_available_gib_sampled']:.1f} GiB | {s['completed_updates_in_metrics']} |")
lines+=['','Comparison contract: 4×L40S; Wan2.2 A14B both experts; 480×832, 33 frames, 20 steps, CFG4.5; 32 videos/update; LoRA r32/alpha64 FP32; native BF16 base with native FP32 exceptions; full FSDP precision_policy none; model offload; no compile; Kling visual_quality; lossless trajectories; parity tolerance exactly zero. Use attempt05/resolved_config.yaml as authoritative.','','For the new staging version, retain the same config, seed, model revisions, precision, sample count, microbatch 1 and correctness gates. Measure a complete update: generation, reward, replay/backward, optimizer, weight sync, checkpoint, and total wall time. Compare warm updates separately from startup. Record exact parity, loss, reward distribution, gradient norm, all-rank success and recoverable checkpoints. Report quality separately using paired held-out prompts/seeds.','','Files: summary.json, gpu_timeseries.csv, batch_timings.csv, raw sampled resources/logs/configs by attempt, source patch/scripts, hardware and package versions. Raw GPU sample rows include process-independent physical VRAM and power. Source patch captures current local changes, not an exact historical source snapshot for attempts 3/4.','']
(out/'README.md').write_text('\n'.join(lines))
save('sha256_manifest.json',{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()})
(a.out/'LATEST').write_text(stamp+'\n');print(out);print('\n'.join(lines[:10]))

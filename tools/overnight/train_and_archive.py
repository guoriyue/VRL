"""Run a configured command and retain receipts (and selected checkpoints) on EBS."""
import argparse,json,shutil,signal,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command',type=Path);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--keep-checkpoints',type=int,default=0);a=p.parse_args()
signal.signal(signal.SIGTERM,lambda *_:None)
child=subprocess.Popen(json.loads(a.command.read_text()))
try: rc=child.wait()
finally:
 a.destination.mkdir(parents=True,exist_ok=True)
 if a.source.exists():
  for item in a.source.iterdir():
   if item.is_file() and item.suffix in {'.json','.jsonl','.csv','.yaml','.log'}: shutil.copy2(item,a.destination/item.name)
  for item in a.source.rglob('*'):
   if item.is_file() and item.suffix in {'.json','.jsonl','.mp4','.png'} and not any(part.startswith('.advantage-spool-') for part in item.parts):
    relative=item.relative_to(a.source)
    if len(relative.parts)>1 and not relative.parts[0].startswith('checkpoint-'):
     target=a.destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item,target)
  checkpoints=sorted((x for x in a.source.glob('checkpoint-*') if x.name.removeprefix('checkpoint-').isdigit() and (x/'checkpoint_meta.json').exists()),key=lambda x:int(x.name.removeprefix('checkpoint-')))
  if a.keep_checkpoints:
   selected=checkpoints[-a.keep_checkpoints:]
   if not selected and (a.source/'checkpoint-final/checkpoint_meta.json').exists(): selected=[a.source/'checkpoint-final']
   for item in selected:
    target=a.destination/item.name
    if not target.exists(): shutil.copytree(item,target)
  (a.destination/'volatile_source.json').write_text(json.dumps({'path':str(a.source),'note':'Remaining source artifacts use local instance storage; selected checkpoints and receipts above are on the root disk'},indent=2))
sys.exit(rc)

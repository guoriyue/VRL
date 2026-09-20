"""Release/recreate only our random H3 fixture if Wan phase parking needs RAM."""
import json,shutil,subprocess,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918';fixture=Path('/dev/shm/vrl-night-20260918/models/h3_random');released=False
while True:
 q=json.loads((out/'queue.json').read_text());job=next(j for j in q['jobs'] if j['id']=='primary_fill_wan13_hpsv3')
 if job['status']!='running':break
 available=psutil.virtual_memory().available
 if available<50*1024**3 and not released and fixture.exists():
  # A pending dependent job cannot be loading this fixture while Wan runs.
  (out/'h3_random_preparation.json').write_text((fixture/'complete.json').read_text())
  (fixture/'complete.json').unlink();shutil.rmtree(fixture);released=True
  (out/'host_headroom_action.json').write_text(json.dumps({'action':'released_own_reproducible_h3_fixture','available_before':available,'time':time.time(),'rebuild_after_wan':True},indent=2))
  print('Released generated H3 fixture for active Wan host headroom',flush=True)
 time.sleep(3)
if released:
 with (out/'h3_rebuild.log').open('w') as log:
  subprocess.run([str(root/'.venv/bin/python'),str(root/'tools/overnight/make_h3_random.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Recreated deterministic H3 fixture',flush=True)

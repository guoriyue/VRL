"""Checksum-verified migration of this session's assets, after their writers exit."""
import argparse,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918'
old=Path('/dev/shm/vrl-night-20260918');new=Path('/mnt/nvme/vrl-night-20260918')
p=argparse.ArgumentParser();p.add_argument('--after-active-job',action='store_true');a=p.parse_args()
assert os.path.ismount('/mnt/nvme')
lock=(out/'nvme-migration.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
def move(relative):
 src=old/relative;dest=new/relative
 if not src.exists() or src.is_symlink():return
 dest.mkdir(parents=True,exist_ok=True)
 subprocess.run(['rsync','-a',str(src)+'/',str(dest)+'/'],check=True)
 verify=subprocess.run(['rsync','-a','--checksum','--dry-run','--itemize-changes',str(src)+'/',str(dest)+'/'],check=True,capture_output=True,text=True)
 assert not verify.stdout.strip(),verify.stdout
 stash=src.with_name(src.name+'.verified-nvme-copy')
 assert not stash.exists()
 src.rename(stash)
 try:src.symlink_to(dest,target_is_directory=True)
 except BaseException:stash.rename(src);raise
 shutil.rmtree(stash)
 with (out/'nvme_migration.jsonl').open('a') as f:f.write(json.dumps({'source':str(src),'destination':str(dest),'verified':'rsync checksum dry-run empty','time':time.time()})+'\n')
 print('Migrated and verified',relative,flush=True)
def wait_terminal(jid):
 while True:
  jobs=json.loads((out/'queue.json').read_text())['jobs'];job=next(j for j in jobs if j['id']==jid)
  if job['status'] not in ['running','ready','pending']:return
  time.sleep(5)
if a.after_active_job:
 wait_terminal('primary_fill_wan13_clean')
 for name in ['HPSv3','Qwen2-VL-7B-Instruct']:move('models/'+name)
 wait_terminal('wan13_clean_health')
 move('runs/wan13_clean')
else:
 for name in ['KlingVideoReward','Qwen2-VL-2B-Instruct']:move('models/'+name)
 move('fixtures/wan13_spool')

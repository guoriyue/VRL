"""Record host and GPU memory while an experiment process exists."""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
pid=int(sys.argv[1]); output=Path(sys.argv[2])
with output.open('a', buffering=1) as f:
 while psutil.pid_exists(pid) and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE:
  rows=[]
  for p in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]:
   try:
    m=p.memory_full_info(); rows.append({'pid':p.pid,'rss':m.rss,'pss':getattr(m,'pss',None),'uss':getattr(m,'uss',None),'cmd':p.cmdline()})
   except psutil.Error: pass
  g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
  f.write(json.dumps({'time':time.time(),'host_available':psutil.virtual_memory().available,'processes':rows,'gpu':g.stdout})+'\n')
  time.sleep(5)

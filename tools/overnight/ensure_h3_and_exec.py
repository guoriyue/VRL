import os
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; refusing root-disk fallback'
"""Build the random H3 fixture only after online Wan has released host memory."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
if not Path('/mnt/nvme/vrl-night-20260918/models/h3_random/complete.json').exists():
 subprocess.run([str(root/'.venv/bin/python'),str(root/'tools/overnight/make_h3_random.py')],check=True)
command=json.loads(Path(sys.argv[1]).read_text());os.execv(command[0],command)

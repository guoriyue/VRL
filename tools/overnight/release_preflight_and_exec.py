"""Prepare the next model on NVMe after H3 passes; retain prior experiment assets."""
import json,os,sys,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/overnight_20260918'
for rank in range(4):
 assert json.loads((out/'h3_fsdp'/f'rank{rank}.json').read_text())['status']=='passed'
source_root=Path('/mnt/nvme/vrl-night-20260918/models')
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; never fall back to root disk'
if not (source_root/'wan22_bf16/preparation_complete.json').exists():
 subprocess.run([str(root/'.venv/bin/python'),str(root/'tools/overnight/prepare_wan22.py')],check=True)
cmd=json.loads(Path(sys.argv[1]).read_text());os.execv(cmd[0],cmd)

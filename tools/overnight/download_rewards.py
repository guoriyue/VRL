import os
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; refusing root-disk fallback'
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path('/mnt/nvme/vrl-night-20260918/models'); root.mkdir(parents=True,exist_ok=True)
for repo, folder, patterns in [('MizzenAI/HPSv3','HPSv3',['HPSv3.safetensors']),('Qwen/Qwen2-VL-7B-Instruct','Qwen2-VL-7B-Instruct',['*.json','*.txt','*.safetensors','*.model'])]:
 print('Downloading',repo,flush=True)
 snapshot_download(repo,local_dir=root/folder,allow_patterns=patterns,max_workers=4)
print('COMPLETE',flush=True)

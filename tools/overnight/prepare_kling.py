import os
assert os.path.ismount('/mnt/nvme'), 'NVMe must be mounted; refusing root-disk fallback'
from pathlib import Path
import json
from huggingface_hub import snapshot_download, HfApi
root=Path('/mnt/nvme/vrl-night-20260918/models'); reward=root/'KlingVideoReward';base=root/'Qwen2-VL-2B-Instruct'
snapshot_download('KlingTeam/VideoReward',revision='4f26600130683e6f1de9f5d463887f28e8ef995c',local_dir=reward,max_workers=2)
revision=HfApi().model_info('Qwen/Qwen2-VL-2B-Instruct').sha
snapshot_download('Qwen/Qwen2-VL-2B-Instruct',revision=revision,local_dir=base,allow_patterns=['*.json','*.txt','*.safetensors','*.model'],max_workers=2)
p=reward/'model_config.json';original=p.read_text();(reward/'model_config.upstream.json').write_text(original)
c=json.loads(original);c['model_config']['model_name_or_path']=str(base);c['model_config']['model_revision']=revision;p.write_text(json.dumps(c,indent=2))
(reward/'preparation_complete.json').write_text(json.dumps({'reward_revision':'4f26600130683e6f1de9f5d463887f28e8ef995c','base_revision':revision,'only_config_change':'local base path and pinned base revision'},indent=2))
print('COMPLETE',flush=True)

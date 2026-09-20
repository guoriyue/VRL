"""Run the existing rollout SP gate with a same-topology repeat control."""
import json, subprocess, sys
from pathlib import Path
from vrl.scripts.perf.sequence_parallel_acceptance import build_parser, _compare
root=Path(__file__).resolve().parents[2]
out=root/'outputs/overnight_20260918/rollout_sp';out.mkdir(parents=True,exist_ok=True)
for name,devices in [('n1',[1]),('n1_repeat',[1]),('n2',[1,2])]:
 cmd=[sys.executable,'-m','vrl.scripts.perf.sequence_parallel_acceptance','generate','--config','experiment/sd3_5/online_grpo_ocr','--gpus-per-engine',str(len(devices)),'--rollout-devices',*[str(d) for d in devices],'--seed','7','--output-dir',str(out/name)]
 with (out/f'{name}.log').open('w') as log: subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
results={}
for name in ['n1_repeat','n2']:
 args=build_parser().parse_args(['compare',str(out/'n1'),str(out/name),'--atol','0.02'])
 results[name]=_compare(args)
(out/'comparison.json').write_text(json.dumps(results,indent=2))
assert all(r['passed'] for r in results.values()),results
(out/'rank0.json').write_text(json.dumps({'status':'passed','comparisons':results},indent=2))

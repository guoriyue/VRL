"""Four-GPU held-out paired base/final-checkpoint evaluation after the night."""
import argparse, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/ubuntu/VRL/.venv/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--night-status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    status=json.loads(a.night_status.read_text());assert status['status']=='passed',status
    checkpoint=Path(status['latest_checkpoint']);run=checkpoint.parent
    prompts=(ROOT/'datasets/videophy_wan22_grpo/val.txt').read_text().splitlines();assert len(prompts)==24
    processes=[]
    env_base=os.environ.copy();env_base.update(PYTHONPATH=f'{ROOT}/.wan_runtime:{ROOT}',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    for mode in ('generate','score'):
        processes=[]
        try:
            for rank in range(4):
                output=Path('/mnt/nvme/vrl-night-20260918/runs/wan14b_final_eval')/f'rank{rank}';output.mkdir(parents=True,exist_ok=True)
                manifest=a.output/f'prompts{rank}.txt';manifest.write_text('\n'.join(prompts[rank*6:(rank+1)*6])+'\n')
                command=[PYTHON,'-m','vrl.scripts.eval.wan22_kling_checkpoint_eval',mode,'--run-dir',str(run),'--output-dir',str(output),'--device','cuda:0']
                if mode=='generate':command+=['--checkpoint',f'final={checkpoint}','--prompts',str(manifest),'--limit','6','--samples-per-prompt','2','--base-seed',str(2026091900+rank*12)]
                env=dict(env_base,CUDA_VISIBLE_DEVICES=str(rank));log=(a.output/f'{mode}{rank}.log').open('w')
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes.append((proc,log))
            deadline=time.time()+7200
            while any(proc.poll() is None for proc,_ in processes):
                failures=[proc.returncode for proc,_ in processes if proc.poll() not in (None,0)]
                if failures:raise RuntimeError(f'{mode} failed: {failures}')
                if time.time()>deadline:raise RuntimeError(f'{mode} exceeded two hours')
                time.sleep(10)
            assert all(proc.returncode==0 for proc,_ in processes)
        finally:
            for proc,log in processes:
                if proc.poll() is None:proc.terminate()
                log.close()
    rows=[]
    for rank in range(4):
        path=Path('/mnt/nvme/vrl-night-20260918/runs/wan14b_final_eval')/f'rank{rank}'/'scores.jsonl'
        local=[json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(local)==24,(rank,len(local))
        for row in local:row['prompt_index']+=rank*6
        rows.extend(local)
    from vrl.scripts.eval.wan22_kling_checkpoint_eval import _prompt_level_paired
    report={'status':'passed','scope':'24 held-out prompts x 2 seeds; paired base vs final, deterministic inference; 96 videos','checkpoint':str(checkpoint),'prompt_level':_prompt_level_paired(rows)}
    (a.output/'scores.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'rank0.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

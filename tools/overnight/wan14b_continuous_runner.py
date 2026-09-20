"""Strict smoke admission, then one warm eight-hour Wan14B training process."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import yaml
from wan14b_night_runner import (ROOT, PYTHON, write_json, check_rows,
                                validate_checkpoint, archive_checkpoint, stop_child)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--run-name',default='continuous')
    args=parser.parse_args()
    attempt=args.attempt.resolve();out=attempt/args.run_name;out.mkdir(exist_ok=True)
    smoke_report=json.loads((attempt/'smoke.acceptance.json').read_text())
    assert smoke_report['status']=='passed'
    checkpoint=Path(smoke_report['checkpoint']);smoke=checkpoint.parent
    check_rows(smoke/'metrics.full_precision.csv')
    for rank in range(4):assert json.loads((smoke/f'training_run_result.rank-{rank}.json').read_text())['status']=='success'
    if args.resume_checkpoint:
        checkpoint=args.resume_checkpoint.resolve()
        check_rows(checkpoint.parent/'metrics.full_precision.csv')
        validate_checkpoint(checkpoint)
    epoch=json.loads((checkpoint/'checkpoint_meta.json').read_text())['next_epoch']
    run=smoke.parent/args.run_name
    assert not run.exists(),f'Refusing to overwrite {run}'
    cfg=yaml.safe_load((attempt/'base.yaml').read_text())
    seconds=int(args.hours*3600)
    cfg['trainer'].update(output_dir=str(run),total_epochs=epoch+100000,
                          max_duration_seconds=seconds,resume_from=str(checkpoint),
                          resume_strict=True,save_freq=1)
    reward=cfg['reward']['kwargs']['kling_video_reward']
    reward.update(archive_dir=str(run/'reward_artifacts'),debug_dir=str(run/'reward_debug'))
    for key in ('artifact_dir','artifact_format','media_type'):reward.pop(key,None)
    cfg['trainer']['replay_parity']['every_update']=True
    assert cfg['trainer']['replay_parity']['max_abs_logprob_diff']==0
    assert cfg['rollout']['trajectory_storage']['dtype']=='preserve'
    config=out/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    from vrl.config.loading import load_config
    from vrl.config.schema import parse_config
    parse_config(load_config(config))
    env=os.environ.copy();env.update(PYTHONPATH=f'{ROOT}/.wan_runtime:{ROOT}',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB='512',RAY_TMPDIR='/mnt/nvme/vrl-night-20260918/ray',
        RAY_memory_usage_threshold='0.95')
    command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
             '--log-dir',str(out/'ranks'),'--redirects','3','--tee','3','-m','vrl.scripts.train','--config',str(config)]
    write_json(out/'command.json',command)
    status={'status':'loading_continuous_training','pid':os.getpid(),'target_seconds':seconds,
            'start_epoch':epoch,'output':str(run),'started':time.time(),
            'scope':'One continuous four-L40S Wan14B run after strict online smoke; parity=0'}
    write_json(out/'status.json',status)
    public=attempt.parent/'status.json';public.unlink();public.symlink_to(out/'status.json')
    child=None;future=None;last_archived=epoch;pool=ThreadPoolExecutor(max_workers=1)
    try:
        with (out/'training.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['training_pid']=child.pid;write_json(out/'status.json',status)
            deadline=time.time()+seconds+14400
            while child.poll() is None:
                if time.time()>deadline:raise RuntimeError('Training exceeded the requested window plus four-hour startup/update allowance')
                metrics=run/'metrics.full_precision.csv'
                if metrics.exists():
                    rows=list(csv.DictReader(metrics.open()))
                    if rows and all(None not in r and None not in r.values() for r in rows):
                        rows=check_rows(metrics)
                        status.update(status='training',completed_new_updates=len(rows),latest_metrics=rows[-1])
                if future and future.done():
                    future.result();future=None
                if future is None and run.exists():
                    choices=[p for p in run.glob('checkpoint-*') if p.name.removeprefix('checkpoint-').isdigit()
                             and int(p.name.removeprefix('checkpoint-'))>last_archived and (p/'checkpoint_meta.json').exists()]
                    if choices:
                        newest=max(choices,key=lambda p:int(p.name.removeprefix('checkpoint-')))
                        last_archived=int(newest.name.removeprefix('checkpoint-'))
                        future=pool.submit(archive_checkpoint,newest,attempt/'persistent_checkpoints')
                        status['archiving_checkpoint']=str(newest)
                write_json(out/'status.json',status)
                time.sleep(15)
            if child.returncode:raise RuntimeError(f'Continuous training exited {child.returncode}')
        rows=check_rows(run/'metrics.full_precision.csv')
        duration=json.loads((run/'training_duration.json').read_text())
        assert duration['stopped_by_duration'] and duration['training_elapsed_seconds']>=seconds,duration
        for rank in range(4):assert json.loads((run/f'training_run_result.rank-{rank}.json').read_text())['status']=='success'
        final=run/'checkpoint-final';experts=validate_checkpoint(final)
        if future:future.result()
        archive_checkpoint(final,attempt/'persistent_checkpoints')
        status.update(status='passed',finished=time.time(),latest_checkpoint=str(final),duration=duration,
                      completed_new_updates=len(rows),experts=experts)
        write_json(out/'status.json',status);write_json(out/'rank0.json',status)
    except BaseException as error:
        if child is not None:stop_child(child)
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished=time.time())
        write_json(out/'status.json',status)
        raise
    finally:
        pool.shutdown(wait=True)


if __name__=='__main__':main()

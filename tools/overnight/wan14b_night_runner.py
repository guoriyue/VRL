"""Run strict Wan14B smoke, then checkpointed training segments for one night.

A segment finishes at an optimizer/checkpoint boundary. Failures stop the queue;
no tolerance changes, implicit retries, or H3 prerequisite substitution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ubuntu/VRL/.venv/bin/python'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def check_rows(path, minimum=1):
    rows = list(csv.DictReader(path.open()))
    if len(rows) < minimum:
        raise RuntimeError(f'Only {len(rows)} completed updates; need {minimum}')
    for row in rows:
        for name in ('loss', 'reward_mean', 'reward_std', 'grad_norm'):
            if not math.isfinite(float(row[name])):
                raise RuntimeError(f'Nonfinite {name}: {row}')
        if float(row['grad_norm']) <= 0 or float(row['reward_std']) <= 0:
            raise RuntimeError(f'No useful gradient or reward variation: {row}')
        if float(row['pre_update_logprob_abs_diff_max']) != 0:
            raise RuntimeError(f'Exact-zero parity failed: {row}')
    return rows


def validate_checkpoint(path):
    import torch
    value = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)
    roots = value['model']['owned_state']
    report = {}
    for name in ('transformer', 'transformer_2'):
        state = roots[name]
        for key, tensor in state.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f'Nonfinite checkpoint tensor {name}.{key}')
        changed = sum(bool(torch.count_nonzero(v)) for k, v in state.items() if 'lora_B' in k)
        if not changed:
            raise RuntimeError(f'Expert {name} has no nonzero LoRA B updates')
        report[name] = {'tensors': len(state), 'nonzero_lora_B_tensors': changed}
    return report


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def archive_checkpoint(source, destination):
    destination.mkdir(exist_ok=True)
    size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if shutil.disk_usage(destination).free < size + 2 * 2**30:
        raise RuntimeError('Insufficient EBS space to preserve the checkpoint plus 2 GiB headroom')
    pending = destination / 'checkpoint-copying'
    if pending.exists():
        raise RuntimeError(f'Unfinished prior checkpoint copy: {pending}')
    shutil.copytree(source, pending)
    previous = destination / 'checkpoint-previous'
    latest = destination / 'checkpoint-latest'
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    pending.rename(latest)
    write_json(destination / 'checkpoint_source.json', {'source': str(source), 'copied_at': time.time()})


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--run-root', type=Path, default=Path('/mnt/nvme/vrl-night-20260918/runs/wan14b_priority'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.json'
    status = {'status': 'waiting_for_model', 'pid': os.getpid(), 'target_hours': args.hours,
              'scope': 'Wan2.2 A14B T2V LoRA; four L40S; Kling visual_quality; exact parity=0'}
    write_json(status_path, status)
    child = None
    try:
        assert os.path.ismount('/mnt/nvme'), 'NVMe is not mounted'
        ray_dir = Path('/mnt/nvme/vrl-night-20260918/ray')
        ray_dir.mkdir(exist_ok=True)
        probe = ray_dir / f'write-test-{os.getpid()}'
        probe.write_text('ok')
        probe.unlink()
        model_ready = Path('/mnt/nvme/vrl-night-20260918/models/wan22_bf16/preparation_complete.json')
        preparation_deadline = time.time() + 7200
        while not model_ready.exists():
            if time.time() >= preparation_deadline:
                raise RuntimeError('Model preparation did not complete within two hours')
            time.sleep(10)
        base = yaml.safe_load(args.config.read_text())
        assert base['trainer']['replay_parity']['max_abs_logprob_diff'] == 0
        assert base['rollout']['trajectory_storage']['dtype'] == 'preserve'
        assert base['rollout']['samples_per_generation_batch'] == base['actor']['training_microbatch_size']
        env = os.environ.copy()
        env.update(PYTHONPATH=f'{ROOT}/.wan_runtime:{ROOT}', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                   VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB='512', RAY_TMPDIR=str(ray_dir),
                   RAY_memory_usage_threshold='0.95')
        resume = None
        epoch = 0
        deadline = None
        stage_number = 0
        last_update_seconds = None
        all_rows = []
        while deadline is None or time.time() < deadline:
            if (args.output / 'STOP_AFTER_SEGMENT').exists():
                status['stop_reason'] = 'requested checkpoint-boundary stop'
                break
            stage = 'smoke' if stage_number == 0 else f'train_{stage_number:03d}'
            count = 1 if stage_number == 0 else 4
            if deadline and last_update_seconds:
                count = min(count, max(1, math.ceil((deadline-time.time()) / last_update_seconds)))
            config = json.loads(json.dumps(base))
            output = args.run_root / stage
            if output.exists():
                raise RuntimeError(f'Refusing to overwrite existing stage {output}')
            config['trainer'].update(output_dir=str(output), total_epochs=epoch+count, save_freq=1)
            if resume:
                config['trainer'].update(resume_from=str(resume), resume_strict=True)
            else:
                config['trainer'].pop('resume_from', None)
            reward = config['reward']['kwargs']['kling_video_reward']
            reward.update(archive_dir=str(output/'reward_artifacts'), debug_dir=str(output/'reward_debug'))
            config_path = args.output / f'{stage}.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                       '--log-dir', str(args.output / f'{stage}_ranks'), '--redirects', '3', '--tee', '3',
                       '-m', 'vrl.scripts.train', '--config', str(config_path)]
            write_json(args.output / f'{stage}.command.json', command)
            started = time.time()
            status.update(status='running_smoke' if stage_number == 0 else 'training', stage=stage,
                          completed_updates=epoch, target_epoch=epoch+count, output=str(output),
                          started_stage=started, training_deadline=deadline)
            with (args.output / f'{stage}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                status['training_pid'] = child.pid
                write_json(status_path, status)
                monitor = subprocess.Popen([PYTHON, str(ROOT/'tools/overnight/monitor.py'),
                                            str(child.pid), str(args.output/f'{stage}.resources.jsonl')],
                                           stdout=log, stderr=subprocess.STDOUT)
                stage_deadline = started + max(14400, count * 3600)
                while child.poll() is None:
                    if time.time() > stage_deadline:
                        raise RuntimeError(f'{stage} exceeded its bounded timeout')
                    metrics = output / 'metrics.full_precision.csv'
                    if metrics.exists() and metrics.stat().st_size:
                        # Writer may be between header and first row. A completed row
                        # is checked immediately, without waiting for the segment.
                        rows = list(csv.DictReader(metrics.open()))
                        if rows and all(None not in row and None not in row.values() for row in rows):
                            check_rows(metrics)
                    time.sleep(10)
                if child.returncode != 0:
                    raise RuntimeError(f'{stage} exited {child.returncode}; see {stage}.log')
            for rank in range(4):
                receipt = json.loads((output/f'training_run_result.rank-{rank}.json').read_text())
                if receipt['status'] != 'success':
                    raise RuntimeError(f'Rank {rank} failed: {receipt}')
            rows = check_rows(output/'metrics.full_precision.csv', minimum=count)
            resume = output / 'checkpoint-final'
            expert_report = validate_checkpoint(resume)
            epoch += count
            write_json(args.output/f'{stage}.acceptance.json', {'status': 'passed', 'rows': rows,
                       'experts': expert_report, 'checkpoint': str(resume)})
            for name in ('metrics.full_precision.csv','training_debug.jsonl','resolved_config.yaml'):
                source = output/name
                if source.exists():
                    shutil.copy2(source, args.output/f'{stage}.{name}')
            archive_checkpoint(resume, args.output/'persistent_checkpoints')
            all_rows.extend(rows)
            last_update_seconds = (time.time()-started) / count
            if deadline is None:
                deadline = time.time() + args.hours*3600
                status['smoke_passed_at'] = time.time()
            stage_number += 1
            status.update(completed_updates=epoch, latest_checkpoint=str(resume), training_deadline=deadline)
            write_json(status_path, status)
            if args.smoke_only:
                break
        if stage_number < 2 and not (args.smoke_only and stage_number == 1):
            raise RuntimeError('Stopped before any sustained-training segment completed')
        status.update(status='passed', finished=time.time(), completed_updates=epoch,
                      note='Smoke acceptance only' if args.smoke_only else 'Stopped after a completed segment/checkpoint; no quality improvement claim')
        write_json(status_path, status)
        write_json(args.output/'rank0.json', status)
    except BaseException as error:
        if child is not None:
            stop_child(child)
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), finished=time.time())
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()

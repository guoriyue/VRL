"""Archive published experiment checkpoints and final results to persistent storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def archive_checkpoint(source: Path, destination: Path) -> None:
    receipt = destination / "archive_sha256.json"
    if receipt.exists():
        return
    size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    if shutil.disk_usage(destination.parent).free < size + 2 * 1024**3:
        raise RuntimeError(f"insufficient archive space for {source}")
    staging = destination.with_name(destination.name + f".archive-{os.getpid()}")
    shutil.copytree(source, staging)
    hashes = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        with path.open("rb") as handle:
            expected = hashlib.file_digest(handle, "sha256").hexdigest()
        with (staging / relative).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"archive checksum mismatch: {relative}")
        hashes[str(relative)] = actual
    (staging / receipt.name).write_text(json.dumps(hashes, indent=2) + "\n")
    staging.rename(destination)
    print(f"Archived and verified {source.name}: {size} bytes", flush=True)


def snapshot_results(source: Path, destination: Path) -> None:
    for name in ("code_snapshot", "eval", "eval_inputs", "logs"):
        source_tree, destination_tree = source / name, destination / name

        def existing_links(
            directory: str,
            names: list[str],
            *,
            source_tree: Path = source_tree,
            destination_tree: Path = destination_tree,
        ) -> list[str]:
            parent = Path(directory)
            target_parent = destination_tree / parent.relative_to(source_tree)
            return [
                item
                for item in names
                if (parent / item).is_symlink()
                and (target_parent / item).is_symlink()
                and os.readlink(parent / item) == os.readlink(target_parent / item)
            ]

        shutil.copytree(
            source_tree,
            destination_tree,
            symlinks=True,
            dirs_exist_ok=True,
            ignore=existing_links,
        )
    for path in source.glob("*.json"):
        shutil.copyfile(path, destination / path.name)
    for pattern in ("*.csv", "*.json", "*.jsonl", "*.yaml"):
        for path in (source / "train").glob(pattern):
            shutil.copyfile(path, destination / "train" / path.name)
    for name in ("admission", "run_evidence"):
        shutil.copytree(
            source / "train" / name,
            destination / "train" / name,
            dirs_exist_ok=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pipeline-pid", type=int, required=True)
    args = parser.parse_args()
    (args.destination / "train").mkdir(parents=True, exist_ok=True)
    process_stat = Path(f"/proc/{args.pipeline_pid}/stat")
    start_time = process_stat.read_text().split()[21]
    pending = {30, 36, 44}
    while True:
        for step in sorted(pending.copy()):
            source = args.source / "train" / f"checkpoint-{step}"
            if (
                step in {36, 44}
                and not (
                    args.source / "adapters" / f"v3e_{step}" / "adapter_model.safetensors"
                ).exists()
            ):
                # Segment endpoints are saved twice; wait until the trainer has
                # exited and the live export starts before copying that tree.
                continue
            # The trainer publishes a fully fsynced staging directory by rename.
            if not (source / "checkpoint_meta.json").exists():
                continue
            archive_checkpoint(source, args.destination / "train" / source.name)
            pending.remove(step)
            snapshot_results(args.source, args.destination)
        try:
            alive = process_stat.read_text().split()[21] == start_time
        except FileNotFoundError:
            alive = False
        try:
            stage = json.loads((args.source / "stage.json").read_text())["stage"]
        except json.JSONDecodeError:
            # The pipeline may be replacing the stage while this poll reads it.
            time.sleep(1)
            continue
        if not alive or stage in {"complete", "failed"}:
            snapshot_results(args.source, args.destination)
            result = {"stage": stage, "pending_checkpoints": sorted(pending)}
            (args.destination / "archive_status.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

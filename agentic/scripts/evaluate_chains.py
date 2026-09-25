"""Run edit chains with the frozen editor and report per-requirement preservation.

Each chain is edited step by step, every state is scored by the configured
judge, the states are exported for independent rescoring, and the scores are
audited against the declared requirements: which step first satisfied each
one, and which step lost it again. This evaluates the editor; it trains nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import torch

from agentic.chains import EditChain, Editor, Judge, load_edit_chains, run_chain
from agentic.export import export_chain_media
from agentic.scripts.session import add_session_arguments, open_session
from vrl.rewards.sequences import EditSequenceSpec, SequenceRequirement, sequence_report
from vrl.run import OnlineRunConfig
from vrl.utils.json_files import canonical_json_sha256, write_json


async def evaluate_chain(
    chain: EditChain,
    editor: Editor,
    judge: Judge,
    requirements: list[SequenceRequirement],
    *,
    output_dir: Path,
    seed: int,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run one chain, export its states, and audit the recorded scores."""

    trace = await run_chain(
        chain, editor, judge, output_dir=output_dir / "run", seed=seed, timeout_s=timeout_s
    )
    exported = export_chain_media(trace, output_dir / "media")
    rows = [
        json.loads(line) for line in (output_dir / "media/media.jsonl").read_text().splitlines()
    ]
    records = {}
    for row, score in zip(rows, trace["state_scores"], strict=True):
        record: dict[str, Any] = {"input": row, "status": "missing"}
        if score is not None:
            record.update(status="success", result={"scores": score["components"]})
        records[row["sample_id"]] = record
    observations = {
        "run_id": canonical_json_sha256(trace, allow_nan=False),
        "config": {"kind": "recorded-judge", "revision": judge.revision},
        "records": records,
    }
    spec = EditSequenceSpec(
        sequence_id=chain.chain_id, samples=exported["sample_order"], requirements=requirements
    )
    report = {
        "schema": "vrl.edit-chain-evaluation.v1",
        "chain_id": chain.chain_id,
        "run_id": observations["run_id"],
        "export_id": exported["export_id"],
        "final_score": trace["final_score"],
        "sequence_report": sequence_report(observations, spec),
    }
    write_json(output_dir / "judge_observations.json", observations)
    write_json(output_dir / "report.json", report)
    return report


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    requirements = [
        SequenceRequirement.model_validate(item)
        for item in json.loads(Path(args.requirements).read_text())
    ]
    chains = load_edit_chains(args.chains, media_dir=output / "chains")
    torch.set_num_threads(8)
    OnlineRunConfig(total_epochs=1, seed=args.seed, deterministic=True).initialize_process_rng()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    write_json(output / "settings.json", vars(args))
    async with open_session(args, output=output) as (editor, judge):
        for chain in chains:
            report = await evaluate_chain(
                chain,
                editor,
                judge,
                requirements,
                output_dir=output / "chains" / chain.chain_id,
                seed=args.seed,
                timeout_s=args.timeout_s,
            )
            audit = report["sequence_report"]
            print(
                json.dumps(
                    {
                        "chain_id": chain.chain_id,
                        "final_score": report["final_score"]["total"],
                        "final_requirements_met": audit["final_requirements_met"],
                        "coverage_complete": audit["coverage_complete"],
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.add_argument("--chains", required=True, type=Path, help="Edit-chain JSONL manifest")
    parser.add_argument(
        "--requirements", required=True, type=Path, help="JSON list of sequence requirements"
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

"""Evaluation runs a chain, exports its states, and reports lost requirements."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agentic.chains import Artifact, ChainRun, EditChain, PolicyStamp, Score
from agentic.scripts.evaluate_chains import evaluate_chain
from agentic.scripts.export_chain_media import main as export_main
from vrl.rewards.evaluation import load_media_manifest
from vrl.rewards.sequences import SequenceRequirement
from vrl.trainers.data.prompts import PromptExample


def _fixture(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "black").save(source)
    chain = EditChain(
        "stages",
        Artifact.from_path(source),
        [
            PromptExample(prompt="Set red channel"),
            PromptExample(prompt="Set green channel; preserve red"),
        ],
        tmp_path / "media",
    )
    parents = []

    async def edit(chain, index, current, *, seed, output_dir):
        parents.append(current)
        # Deliberate fixture defect: the second edit forgets the first channel.
        path = output_dir / f"step{index}.png"
        Image.new("RGB", (4, 4), (255, 0, 0) if index == 0 else (0, 255, 0)).save(path)
        return Artifact.from_path(path)

    async def score(chain, artifacts):
        scores = []
        for artifact in artifacts:
            with Image.open(artifact.path) as image:
                r, g, _ = image.getpixel((0, 0))
            scores.append(Score((r + g) / 510, {"red": r / 255, "green": g / 255}))
        return scores

    editor = SimpleNamespace(
        policy_stamp=PolicyStamp("fixture-editor", "v1", 0),
        activate=AsyncMock(),
        park=AsyncMock(),
        edit=edit,
    )
    judge = SimpleNamespace(
        revision="pixel-fixture", activate=AsyncMock(), park=AsyncMock(), score=score
    )
    return chain, editor, judge, parents


@pytest.mark.asyncio
async def test_report_names_the_step_that_lost_an_earlier_requirement(tmp_path):
    chain, editor, judge, parents = _fixture(tmp_path)
    requirements = [
        SequenceRequirement(name="red", axis="red", threshold=1.0, active_from=1),
        SequenceRequirement(name="green", axis="green", threshold=1.0, active_from=2),
    ]
    output = tmp_path / "eval"
    report = await evaluate_chain(chain, editor, judge, requirements, output_dir=output, seed=1)
    trace = json.loads((output / "run/run.json").read_text())
    assert parents[0] == chain.source
    assert parents[1].sha256 == trace["steps"][0]["artifact"]["sha256"]
    audit = report["sequence_report"]
    assert audit["requirements"]["red"]["regression_steps"] == [2]
    assert audit["requirements"]["green"]["first_satisfied_step"] == 2
    assert audit["final_requirements_met"] is False and audit["coverage_complete"]
    assert report["final_score"]["total"] == pytest.approx(0.5)
    provenance = json.loads((output / "media/provenance.json").read_text())
    assert provenance["sample_order"] == [
        "stages:state:0000",
        "stages:state:0001",
        "stages:state:0002",
    ]
    assert provenance["lineage"][2]["parent_sha256"] == parents[1].sha256
    assert json.loads((output / "report.json").read_text()) == report

    editor.edit = AsyncMock(side_effect=RuntimeError("tool failed"))
    with pytest.raises(RuntimeError, match="tool failed"):
        await evaluate_chain(
            chain, editor, judge, requirements, output_dir=tmp_path / "bad", seed=1
        )
    assert not (tmp_path / "bad/report.json").exists()
    assert json.loads((tmp_path / "bad/run/run.json").read_text())["status"] == "error"


@pytest.mark.asyncio
async def test_export_cli_lists_source_then_outputs_with_the_source_as_reference(tmp_path):
    chain, editor, judge, _ = _fixture(tmp_path)
    run = await chain.run(editor, judge, output_dir=tmp_path / "run")
    export_main(
        ["--run", str(tmp_path / "run/run.json"), "--output-dir", str(tmp_path / "export")]
    )
    rows = load_media_manifest(tmp_path / "export/media.jsonl")
    assert [row.path.name for row in rows] == ["source.png", "step0.png", "step1.png"]
    assert all(row.assets["reference_image"] == tmp_path / "source.png" for row in rows)
    assert rows[0].metadata == rows[2].metadata
    with pytest.raises(FileExistsError):
        run.export(tmp_path / "export")
    with pytest.raises(ValueError, match="successful"):
        ChainRun({**run.record, "status": "error"}).export(tmp_path / "failed")

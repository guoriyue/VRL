"""The reward card: one reward's qualification evidence, gate by gate.

A reward enters a training key only when the card says every pre-training
gate passed for the axis being optimised:

1. **repeatability** -- repeat scoring of identical inputs moves the axis by at
   most ``repeat_tolerance`` (``python -m reward repeat``);
2. **agreement** -- each declared label contrast is ranked at AUC >= the
   contrast gate (``python -m reward agreement``); a verifiable reward with an
   oracle set uses the same report with the oracle verdict as the label;
3. **shortcuts** -- no stress or shortcut transform scores at or above the
   genuine candidate on more than ``shortcut_tolerance`` of sources
   (``python -m reward stress`` over ``stress-manifest`` / ``shortcut-manifest``);
4. **spread** -- under the training sampler the success rate sits inside the
   band and enough prompt groups carry within-group differences
   (``python -m reward spread``).

The post-training comparison (gate 5: held-out reward delta vs blind verdicts
per checkpoint) is recorded on the card as free-form evidence; the tool does
not judge it, because "reward up, judges flat" is a decision, not a number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vrl.utils.json_files import write_json

CARD_SCHEMA = "vrl.reward-card.v1"


def _read(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def build_card(
    *,
    name: str,
    kind: str,
    axis: str,
    repeat: Path | None = None,
    agreement: Path | None = None,
    stress: Path | None = None,
    spread: Path | None = None,
    post_training: Path | None = None,
    repeat_tolerance: float = 1e-6,
    shortcut_tolerance: float = 0.1,
    min_mixed_share: float = 0.3,
    blind_spots: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the gate verdicts from the individual reports (missing report = gate not run)."""

    if kind not in ("verifiable", "learned"):
        raise ValueError("kind must be 'verifiable' or 'learned'")
    gates: dict[str, dict[str, Any]] = {}

    report = _read(repeat)
    if report is None:
        gates["repeatability"] = {"passed": None, "reason": "not run"}
    else:
        axis_report = report["axes"].get(axis)
        if axis_report is None:
            gates["repeatability"] = {"passed": False, "reason": f"axis {axis!r} absent"}
        else:
            worst = axis_report["max_score_range"]
            gates["repeatability"] = {
                "passed": worst is not None and worst <= repeat_tolerance,
                "max_score_range": worst,
                "tolerance": repeat_tolerance,
                "repeat_count": report["repeat_count"],
                "sample_count": report["sample_count"],
            }

    report = _read(agreement)
    if report is None:
        gates["agreement"] = {"passed": None, "reason": "not run"}
    else:
        if report["axis"] != axis:
            raise ValueError(f"agreement report is for axis {report['axis']!r}, card is {axis!r}")
        verdicts = {
            contrast: {
                "passed": entry["passed"],
                "auc": entry["auc"],
                "auc_bootstrap_95ci": entry["auc_bootstrap_95ci"],
                "positive_count": entry["positive_count"],
                "negative_count": entry["negative_count"],
            }
            for contrast, entry in report["contrasts"].items()
        }
        gates["agreement"] = {
            "passed": bool(verdicts) and all(v["passed"] is True for v in verdicts.values()),
            "gate": report["gate"],
            "contrasts": verdicts,
            "unscored": len(report["unscored_sample_ids"]),
        }

    report = _read(stress)
    if report is None:
        gates["shortcuts"] = {"passed": None, "reason": "not run"}
    else:
        transforms = {}
        for transform, axes in report["transforms"].items():
            entry = axes.get(axis)
            if entry is None:
                transforms[transform] = {"passed": False, "reason": f"axis {axis!r} absent"}
                continue
            total = entry["increases"] + entry["decreases"] + entry["ties"]
            not_below = (entry["increases"] + entry["ties"]) / total if total else None
            transforms[transform] = {
                "passed": not_below is not None and not_below <= shortcut_tolerance,
                "scored_at_or_above_genuine_share": not_below,
                "sources": total,
                "mean_delta": entry["paired_delta"]["mean"],
            }
        gates["shortcuts"] = {
            "passed": bool(transforms) and all(t["passed"] is True for t in transforms.values()),
            "tolerance": shortcut_tolerance,
            "transforms": transforms,
        }

    report = _read(spread)
    if report is None:
        gates["spread"] = {"passed": None, "reason": "not run"}
    else:
        if report["axis"] != axis:
            raise ValueError(f"spread report is for axis {report['axis']!r}, card is {axis!r}")
        gates["spread"] = {
            "passed": bool(report["in_band"]) and report["mixed_prompt_share"] >= min_mixed_share,
            "success_rate": report["success_rate"],
            "band": report["band"],
            "mixed_prompt_share": report["mixed_prompt_share"],
            "min_mixed_share": min_mixed_share,
            "mean_within_prompt_std": report["mean_within_prompt_std"],
            "zero_spread_prompt_share": report["zero_spread_prompt_share"],
        }

    verdicts = [gate["passed"] for gate in gates.values()]
    return {
        "schema": CARD_SCHEMA,
        "name": name,
        "kind": kind,
        "axis": axis,
        "gates": gates,
        "blind_spots": list(blind_spots or []),
        "post_training": _read(post_training),
        "ready_for_training_key": all(v is True for v in verdicts),
        "gates_not_run": [gate for gate, value in gates.items() if value["passed"] is None],
    }


def render_markdown(card: dict[str, Any]) -> str:
    """A reader-facing card: verdict first, then one block per gate."""

    def mark(value: Any) -> str:
        return "PASS" if value is True else "FAIL" if value is False else "not run"

    lines = [
        f"# Reward card: {card['name']}",
        "",
        f"- kind: {card['kind']}",
        f"- axis: `{card['axis']}`",
        f"- ready for a training key: **{'yes' if card['ready_for_training_key'] else 'no'}**"
        + (f" (not run: {', '.join(card['gates_not_run'])})" if card["gates_not_run"] else ""),
    ]
    if card["blind_spots"]:
        lines.append("- blind spots: " + "; ".join(card["blind_spots"]))
    gates = card["gates"]
    lines += ["", "## 1. Repeatability -- " + mark(gates["repeatability"]["passed"])]
    if "max_score_range" in gates["repeatability"]:
        g = gates["repeatability"]
        lines.append(
            f"max score range {g['max_score_range']} (tolerance {g['tolerance']}) over "
            f"{g['sample_count']} samples x {g['repeat_count']} repeats"
        )
    lines += ["", "## 2. Agreement with labels -- " + mark(gates["agreement"]["passed"])]
    for contrast, entry in gates["agreement"].get("contrasts", {}).items():
        ci = entry["auc_bootstrap_95ci"]
        auc = "n/a" if entry["auc"] is None else f"{entry['auc']:.3f}"
        band = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"- {contrast}: AUC {auc}{band} "
            f"({entry['positive_count']} vs {entry['negative_count']}) -- {mark(entry['passed'])}"
        )
    lines += ["", "## 3. Shortcuts and damage -- " + mark(gates["shortcuts"]["passed"])]
    for transform, entry in gates["shortcuts"].get("transforms", {}).items():
        share = entry.get("scored_at_or_above_genuine_share")
        shown = "n/a" if share is None else f"{share:.0%}"
        lines.append(
            f"- {transform}: scored at or above the genuine output on {shown} of "
            f"{entry.get('sources', 0)} sources -- {mark(entry['passed'])}"
        )
    lines += ["", "## 4. Spread under the training sampler -- " + mark(gates["spread"]["passed"])]
    if "success_rate" in gates["spread"]:
        g = gates["spread"]
        lines.append(
            f"success rate {g['success_rate']:.0%} (band {g['band'][0]:.0%}-{g['band'][1]:.0%}); "
            f"prompts with mixed outcomes {g['mixed_prompt_share']:.0%} (min {g['min_mixed_share']:.0%}); "
            f"mean within-prompt std {g['mean_within_prompt_std']:.3f}; "
            f"zero-spread prompts {g['zero_spread_prompt_share']:.0%}"
        )
    lines += ["", "## 5. After training"]
    post = card["post_training"]
    lines.append(
        "not yet trained on" if post is None else json.dumps(post, ensure_ascii=False, indent=1)
    )
    return "\n".join(lines) + "\n"


def write_card(card: dict[str, Any], path: Path) -> None:
    write_json(path, card)
    path.with_suffix(".md").write_text(render_markdown(card), encoding="utf-8")


__all__ = ["build_card", "render_markdown", "write_card"]

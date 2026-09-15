"""Per-block ``torch.compile`` of a policy core's repeated transformer blocks.

A DiT is one short stem, N identical blocks, and one head. Compiling the root
module traces and lowers all N blocks as one graph, so cold compile time grows
with N (tens of minutes for a 512p Cosmos shape, see
``docs/sprints/done/SPRINT_ray_rollout_operation_deadlines.md``) and every
recompile — a new resolution, a partial last batch — pays the whole graph
again. Compiling each block in place instead lowers the block's code once:
dynamo treats block parameters as graph inputs, so every instance of the block
class shares the one cache entry as long as its shapes and dtypes match. The
stem and head stay eager; they are a handful of launches per step, and the only
fusion lost is across the block boundary (one residual add per block).

The compiled module tree keeps its type and its ``state_dict`` namespace —
``nn.Module.compile`` swaps the call path, it does not wrap — so weight sync
and checkpoints see the plain policy namespace and nothing needs peeling.
"""

from __future__ import annotations

import torch.nn as nn


def repeated_block_lists(root: nn.Module) -> dict[str, nn.ModuleList]:
    """Outermost ``nn.ModuleList``s of two or more modules of one class.

    Keyed by qualified name under ``root``. Walks the whole tree, not only
    direct children, because a LoRA-adapted policy nests the transformer under
    ``base_model.model``. A list found inside an already selected list is not
    returned: compiling the outer block covers it.
    """

    selected: dict[str, nn.ModuleList] = {}
    for name, module in root.named_modules():
        if not name or not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        if any(name.startswith(f"{outer}.") for outer in selected):
            continue
        first = type(module[0])
        if all(type(block) is first for block in module):
            selected[name] = module
    return selected


def compile_repeated_blocks(root: nn.Module, *, mode: str) -> int:
    """Compile every block of every repeated block list in place; return the count.

    Raises when ``root`` has no repeated block list: a policy with nothing to
    compile per block must fail loudly rather than run eager under a config
    that asked for compilation.
    """

    lists = repeated_block_lists(root)
    if not lists:
        raise ValueError(
            f"{type(root).__name__} has no repeated block list (an nn.ModuleList of "
            "two or more same-class modules); regional compile has nothing to compile",
        )
    count = 0
    for blocks in lists.values():
        for block in blocks:
            block.compile(mode=mode, fullgraph=False)
            count += 1
    return count


def compiled_block_count(root: nn.Module) -> int:
    """How many modules under ``root`` run through an in-place compiled call."""

    return sum(
        getattr(module, "_compiled_call_impl", None) is not None for module in root.modules()
    )

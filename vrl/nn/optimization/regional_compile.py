"""The repeated transformer blocks of a policy core.

A DiT is one short stem, N identical blocks, and one head. Host streaming
(``vrl.nn.offload.block_streaming``) keeps the frozen weights of those blocks
on the host and streams them per call, so it needs the block lists found here.
"""

from __future__ import annotations

import torch.nn as nn


def repeated_block_lists(root: nn.Module) -> dict[str, nn.ModuleList]:
    """Outermost ``nn.ModuleList``s of two or more modules of one class.

    Keyed by qualified name under ``root``. Walks the whole tree, not only
    direct children, because a LoRA-adapted policy nests the transformer under
    ``base_model.model``. A list found inside an already selected list is not
    returned: streaming the outer block covers it.
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

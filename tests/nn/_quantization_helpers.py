"""The quantizable rollout-policy double shared by the nn optimization and quantization tests."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class QuantizablePolicy:
    """A rollout policy whose declared cores the real quantization swap can match.

    Each root in ``roots`` holds ``ff.<name>`` for every name in ``linears``: an
    MLP-path ``Linear(1024, 1024)``, so it sits inside the NVFP4 ``mlp_only``
    profile and clears ``min_features``. No ``linears`` is a policy with nothing
    quantizable, the no-op case the pass must reject.
    """

    quantization_exclude: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        roots: Sequence[str] = ("transformer",),
        linears: Sequence[str] = ("net",),
    ) -> None:
        self._cores: dict[str, nn.Module] = {}
        for root_name in roots:
            root = nn.Module()
            root.ff = nn.Module()
            for name in linears:
                setattr(root.ff, name, nn.Linear(1024, 1024, bias=False))
            self._cores[root_name] = root

    @property
    def policy_cores(self) -> dict[str, nn.Module]:
        return self._cores

    def swapped(self, scheme: type[nn.Module]) -> list[nn.Module]:
        """The ``scheme`` modules now in the cores: the swap's effect, not its call."""

        return [
            module
            for core in self._cores.values()
            for module in core.modules()
            if isinstance(module, scheme)
        ]

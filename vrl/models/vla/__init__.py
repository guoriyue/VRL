"""Vision-language-action (robotics) model seam.

Distinct from ``vrl/models/diffusion`` and ``vrl/models/ar``: a VLA policy's
output is an action chunk consumed by a closed-loop environment, not an
image/video latent. See ``vrl/models/vla/policy.py`` for the contract.
"""

from vrl.models.vla.policy import ActionPolicy

__all__ = ["ActionPolicy"]

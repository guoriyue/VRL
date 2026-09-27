"""The local-edit red-box hint: one string shared by the manifest and the reward.

The data script appends it to a prompt whose reference image carries a drawn red
box, and the local-edit reward strips it again before asking the execution
judge, which must see the plain instruction. Both sides read this constant so
the two cannot drift; it lives with the reward assets because the reward is the
lower layer and the data script may depend on it, not the other way round.
"""

from __future__ import annotations

HINT_SUFFIX = " Edit only inside the red box, then remove the red box."

__all__ = ["HINT_SUFFIX"]

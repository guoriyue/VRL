"""Public facade for Video2World dataset preparation."""

# Preserve named attributes from the flat module without widening its historical
# wildcard-import surface.
from vrl.scripts.data.video_world.cli import (  # noqa: F401
    COMMAND_NAME,
    DATASET_NAME,
    TARGET_COMMAND_NAME,
    manifest_setup_hints,
    register,
)
from vrl.scripts.data.video_world.manifests import (
    build_target_video_world_rows,
    build_video_world_rows,
)

__all__ = [
    "build_target_video_world_rows",
    "build_video_world_rows",
    "manifest_setup_hints",
    "register",
]

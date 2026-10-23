"""Public facade for Video2World dataset preparation."""

from vrl.scripts.data.video_world.dataset_index import (
    build_target_video_world_rows,
    build_video_world_rows,
)

__all__ = [
    "build_target_video_world_rows",
    "build_video_world_rows",
]

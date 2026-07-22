"""Tests for deriving Wan T2V rows from target-backed Video2World data."""

from __future__ import annotations

from vrl.scripts.data.derive_text_video_targets import derive_text_video_rows


def test_derive_text_video_rows_removes_unconsumed_first_frame() -> None:
    rows = derive_text_video_rows(
        [
            {
                "prompt": "Put the blue block in the bowl",
                "reference_image": "video_world/references/episode.png",
                "target_video": "video_world/targets/episode.mp4",
                "task_type": "video2world",
                "metadata": {
                    "source": "droid",
                    "source_episode": "000001",
                    "conditioning": "first_frame",
                },
            },
        ],
    )

    assert rows == [
        {
            "prompt": "Put the blue block in the bowl",
            "target_video": "video_world/targets/episode.mp4",
            "task_type": "text_to_video",
            "metadata": {
                "source": "droid",
                "source_episode": "000001",
                "source_conditioning": "first_frame",
                "conditioning": "text_only",
                "derived_from_task_type": "video2world",
            },
        },
    ]

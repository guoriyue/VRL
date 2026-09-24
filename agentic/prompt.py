"""The controller's input text, rendered from task fields (the Edit-R1 pattern).

The template is the only prompt in the package; everything it shows comes
from the task row and the observation: the instruction, an optional
requirement, any named context images the task declares, the previous action
and the remaining budget. Changing this text changes the policy, so it is part
of the controller's base identity.
"""

from __future__ import annotations

from agentic.episode import Observation, Task

CONTROLLER = """{images}
Instruction: {instruction}
Requirements: {requirement}
Previous action: {previous_action}
Remaining editing calls: {remaining_tool_calls}
Choose the next action. Reply with its letter only.
{choices}"""

STOP = "Stop editing."

ALPHA = (
    "the alpha mask of image {index} (white opaque, black transparent, gray partial); "
    "it describes that image, not a target"
)


def image_labels(task: Task, *, alpha: bool) -> list[str]:
    """What each image the controller receives is, in the order they are sent."""

    labels = ["the original image", "the current result"]
    labels += [f"{name} (context)" for name in task.context_images]
    if alpha:
        labels += [ALPHA.format(index=1), ALPHA.format(index=2)]
    return labels


def render(task: Task, observation: Observation, *, alpha: bool) -> str:
    labels = image_labels(task, alpha=alpha)
    letters = [chr(ord("A") + index) for index in range(len(task.actions))]
    return CONTROLLER.format(
        images="\n".join(f"Image {index + 1} is {label}." for index, label in enumerate(labels)),
        instruction=task.instruction,
        requirement=task.requirement or "none",
        previous_action=observation.previous_action or "none",
        remaining_tool_calls=observation.remaining_tool_calls,
        choices="\n".join(
            f"{letter}: {action.instruction if action.kind == 'edit' else STOP}"
            for letter, action in zip(letters, task.actions, strict=True)
        ),
    )


__all__ = ["ALPHA", "CONTROLLER", "STOP", "image_labels", "render"]

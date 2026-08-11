"""Checkpoint-owned reward assets packaged into wheels.

This package holds data that belongs to a reward checkpoint rather than to any
workflow module: prompt templates and parser grammars the judge models were
trained/evaluated with (``kling_prompt_templates``, ``video_judge_prompts``)
and small weight files loaded via ``importlib.resources``
(``sac+logos+ava1-l14-linearMSE.pth`` for the aesthetic head; see the
``vrl.rewards.assets`` package-data entry in ``pyproject.toml``). Keeping the
taxonomy here keeps model loaders free of large business-prompt constants.
"""

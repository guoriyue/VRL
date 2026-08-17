"""HPSv3 scoring prompt, copied verbatim from MizzenAI/HPSv3 (MIT).

Source: ``hpsv3/dataset/data_collator_qwen.py`` (``INSTRUCTION`` and
``prompt_with_special_token``) at branch ``upgrade_transformers_version``.
The strings are byte-exact copies extracted mechanically from the upstream
module: the checkpoint was trained against this exact wording and whitespace,
so any edit silently shifts the score distribution.
"""

from __future__ import annotations

HPSV3_REWARD_SPECIAL_TOKEN = "<|Reward|>"

HPSV3_INSTRUCTION = "\nYou are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n\n**Visual Quality:**  \nEvaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n\n**Text Alignment:**  \nAssess how well the image matches the textual prompt across the following sub-dimensions:\n- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.\n- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.\n- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.\n- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.\n- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.\nTextual prompt - {text_prompt}\n\n\n"

HPSV3_SPECIAL_TOKEN_SUFFIX = (
    "\nPlease provide the overall ratings of this image: <|Reward|>\n\nEND\n"
)


def build_hpsv3_frame_prompt(text_prompt: str) -> str:
    """The full user-turn text for scoring one frame against ``text_prompt``."""

    return HPSV3_INSTRUCTION.format(text_prompt=text_prompt) + HPSV3_SPECIAL_TOKEN_SUFFIX


__all__ = [
    "HPSV3_INSTRUCTION",
    "HPSV3_REWARD_SPECIAL_TOKEN",
    "HPSV3_SPECIAL_TOKEN_SUFFIX",
    "build_hpsv3_frame_prompt",
]

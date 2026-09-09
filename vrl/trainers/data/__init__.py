"""Trainer data utilities.

Lazy public boundary: the dataset/collate modules pull ``torch.utils.data``, but
``prompt_sampler.PromptSamplingStrategy`` is a plain Enum that
``vrl.config.schema`` reads while validating every recipe. An eager re-export
here charged all config parsing for the Pick-a-Pic and prompt datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrl.utils.config import install_lazy_exports

if TYPE_CHECKING:
    from vrl.trainers.data.artifacts import DatasetFileReport as DatasetFileReport
    from vrl.trainers.data.artifacts import ResolvedArtifact as ResolvedArtifact
    from vrl.trainers.data.artifacts import (
        resolve_prompt_example_artifacts as resolve_prompt_example_artifacts,
    )
    from vrl.trainers.data.artifacts import (
        resolve_prompt_example_references as resolve_prompt_example_references,
    )
    from vrl.trainers.data.preferences import (
        PickAPicPreferenceDataset as PickAPicPreferenceDataset,
    )
    from vrl.trainers.data.preferences import PreferenceBatch as PreferenceBatch
    from vrl.trainers.data.prompt_sampler import PromptBatchSampler as PromptBatchSampler
    from vrl.trainers.data.prompt_sampler import PromptSamplingStrategy as PromptSamplingStrategy
    from vrl.trainers.data.prompts import ImageCaptionPromptDataset as ImageCaptionPromptDataset
    from vrl.trainers.data.prompts import JsonlPromptDataset as JsonlPromptDataset
    from vrl.trainers.data.prompts import PromptExample as PromptExample
    from vrl.trainers.data.prompts import load_prompt_dataset_index as load_prompt_dataset_index
    from vrl.trainers.data.prompts import (
        load_prompt_examples_from_config as load_prompt_examples_from_config,
    )
    from vrl.trainers.data.prompts import load_prompt_image_manifest as load_prompt_image_manifest

_PUBLIC_EXPORTS = {
    "DatasetFileReport": "vrl.trainers.data.artifacts",
    "ImageCaptionPromptDataset": "vrl.trainers.data.prompts",
    "JsonlPromptDataset": "vrl.trainers.data.prompts",
    "PickAPicPreferenceDataset": "vrl.trainers.data.preferences",
    "PreferenceBatch": "vrl.trainers.data.preferences",
    "PromptBatchSampler": "vrl.trainers.data.prompt_sampler",
    "PromptExample": "vrl.trainers.data.prompts",
    "PromptSamplingStrategy": "vrl.trainers.data.prompt_sampler",
    "ResolvedArtifact": "vrl.trainers.data.artifacts",
    "load_prompt_examples_from_config": "vrl.trainers.data.prompts",
    "load_prompt_image_manifest": "vrl.trainers.data.prompts",
    "load_prompt_dataset_index": "vrl.trainers.data.prompts",
    "DatasetProvenance": "vrl.trainers.data.provenance",
    "resolve_prompt_example_artifacts": "vrl.trainers.data.artifacts",
    "resolve_prompt_example_references": "vrl.trainers.data.artifacts",
}

install_lazy_exports(globals(), _PUBLIC_EXPORTS)

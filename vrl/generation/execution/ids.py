"""Generation sample identity helpers."""

from __future__ import annotations

from vrl.generation.types import GenerationRequest, GenerationSampleRow


class GenerationIdFactory:
    """Build deterministic sample rows from a generation request."""

    def build_sample_rows(
        self,
        request: GenerationRequest,
    ) -> list[GenerationSampleRow]:
        base_seed = request.sampling.get("seed")
        seed_int = int(base_seed) if base_seed is not None else None
        rows: list[GenerationSampleRow] = []
        for prompt_index, prompt in enumerate(request.prompts):
            prompt_id = f"{request.request_id}:prompt:{prompt_index}"
            group_id = prompt_id
            for sample_index in range(request.samples_per_prompt):
                flat_index = len(rows)
                sample_id = f"{prompt_id}:sample:{sample_index}"
                metadata = dict(request.metadata)
                metadata.update(
                    {
                        "request_id": request.request_id,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "flat_sample_index": flat_index,
                        "policy_version": request.policy_version,
                    }
                )
                rows.append(
                    GenerationSampleRow(
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        prompt=prompt,
                        prompt_id=prompt_id,
                        group_id=group_id,
                        sample_id=sample_id,
                        trajectory_id=sample_id,
                        seed=None if seed_int is None else seed_int + flat_index,
                        metadata=metadata,
                    )
                )
        return rows


__all__ = ["GenerationIdFactory"]

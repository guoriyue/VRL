"""Model-boundary token sharding for the Cosmos transformer block stack."""

from contextlib import contextmanager
from inspect import signature

import torch.distributed as dist

from vrl.models.families.cosmos import CosmosContextParallelSelfAttnProcessor
from vrl.trainers.distributed import context_parallel_gather_tokens


@contextmanager
def cosmos_context_parallel(transformer, *, group):
    """Keep block activations sharded and return replicated full model outputs.

    Keep this context active through backward, including checkpoint recompute.
    Each rank must have identical inputs, weights and collective ordering.
    Divide replicated full-output losses by CP size and SUM parameter gradients
    across CP. This installs neither that strategy nor the BF16 precision policy.
    Not compatible with concurrent forwards or an already sharded transformer.
    """
    if not dist.is_initialized():
        raise RuntimeError("Cosmos CP requires an initialized process group")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if world < 2 or rank < 0:
        raise ValueError("Cosmos CP requires membership in a group of >= 2")
    blocks = transformer.transformer_blocks
    if not blocks:
        raise ValueError("Cosmos CP requires transformer blocks")
    for block in blocks:
        if isinstance(block.attn1.processor, CosmosContextParallelSelfAttnProcessor):
            raise ValueError("Cosmos CP is already installed")
        if block.attn1.heads % world:
            raise ValueError("Cosmos attention heads must be divisible by CP size")
        if block.before_proj is not None or block.after_proj is not None:
            raise ValueError("Cosmos CP does not support ControlNet projection blocks")

    def split(tensor, dim):
        if tensor.shape[dim] == 0 or tensor.shape[dim] % world:
            raise ValueError("Cosmos token count must be nonempty and divisible by CP size")
        return tensor.chunk(world, dim=dim)[rank].contiguous()

    def token_condition(tensor):
        if tensor is not None and tensor.ndim == 3 and tensor.shape[1] > 1:
            return split(tensor, 1)
        return tensor

    def block_hook(block, first):
        forward_signature = signature(block.forward)

        def prepare(_module, args, kwargs):
            bound = forward_signature.bind(*args, **kwargs)
            values = bound.arguments
            if first:
                values["hidden_states"] = split(values["hidden_states"], 1)
            for name in ("embedded_timestep", "temb", "extra_pos_emb", "controlnet_residual"):
                if name in values:
                    values[name] = token_condition(values[name])
            rope = values.get("image_rotary_emb")
            if rope is not None:
                values["image_rotary_emb"] = tuple(split(tensor, 0) for tensor in rope)
            return bound.args, bound.kwargs

        return prepare

    norm_signature = signature(transformer.norm_out.forward)

    def prepare_norm(_module, args, kwargs):
        bound = norm_signature.bind(*args, **kwargs)
        for name in ("embedded_timestep", "temb"):
            if name in bound.arguments:
                bound.arguments[name] = token_condition(bound.arguments[name])
        return bound.args, bound.kwargs

    def gather_output(_module, _args, output):
        return context_parallel_gather_tokens(output, group=group)

    handles, processors = [], []
    try:
        for index, block in enumerate(blocks):
            processors.append((block.attn1, block.attn1.processor))
            block.attn1.set_processor(CosmosContextParallelSelfAttnProcessor(group))
            handles.append(
                block.register_forward_pre_hook(block_hook(block, index == 0), with_kwargs=True)
            )
        handles.append(
            transformer.norm_out.register_forward_pre_hook(prepare_norm, with_kwargs=True)
        )
        handles.append(transformer.proj_out.register_forward_hook(gather_output))
        yield transformer
    finally:
        for handle in handles:
            handle.remove()
        for attention, processor in processors:
            attention.set_processor(processor)

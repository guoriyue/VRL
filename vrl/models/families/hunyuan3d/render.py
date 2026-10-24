"""Occupancy-grid decode and rendering for the Hunyuan3D shape family.

The ShapeVAE's geometry decoder answers "is this point inside?" as a logit per
query point. Upstream turns a dense grid of those logits into a mesh with
marching cubes; the RL loop instead needs an image tensor a frame-based reward
can score, so this module queries the same dense grid and renders it directly:
orthographic rays through the volume, the first sample whose interpolated logit
crosses zero is the surface, and the shaded normal there is the pixel. No mesh
library is involved and everything stays a torch tensor on the model device.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Upstream's query-box half-width (``box_v``): the latent space is normalized so
# shapes fit in [-1, 1]; the margin keeps the surface off the grid border.
BOUNDS = 1.01
# Upstream's decision level for the occupancy logits (``mc_level``).
SURFACE_LEVEL = 0.0


def dense_grid_points(resolution: int, *, bounds: float, device: torch.device) -> torch.Tensor:
    """``[(R + 1)^3, 3]`` corner points of a ``resolution``-cell cube, ``ij`` ordered
    (x slowest) exactly as upstream's ``generate_dense_grid_points``."""

    axis = torch.linspace(-bounds, bounds, resolution + 1, device=device, dtype=torch.float32)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((xs, ys, zs), dim=-1).reshape(-1, 3)


def query_occupancy_grid(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    resolution: int,
    bounds: float = BOUNDS,
    chunk: int = 65536,
) -> torch.Tensor:
    """Occupancy logits of ``latents`` on the dense grid, ``[B, R + 1, R + 1, R + 1]`` fp32.

    ``latents`` are the sampled DiT latents; this applies the VAE's inverse
    scale, its latent transformer, then the chunked geometry-decoder queries
    (upstream's ``_export`` plus ``VanillaVolumeDecoder``).
    """

    batch = latents.shape[0]
    decoded = vae(latents.to(vae.post_kl.weight.dtype) / vae.scale_factor)
    points = dense_grid_points(resolution, bounds=bounds, device=latents.device)
    logits = []
    for start in range(0, points.shape[0], chunk):
        queries = points[start : start + chunk].to(decoded.dtype)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        logits.append(vae.geo_decoder(queries=queries, latents=decoded).float())
    side = resolution + 1
    return torch.cat(logits, dim=1).view(batch, side, side, side)


def _view_basis(
    azimuth: float, elevation: float, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Camera axes for a viewer on the unit sphere: (right, up, towards the shape)."""

    forward = -torch.tensor(
        [
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ],
        device=device,
    )
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = F.normalize(torch.linalg.cross(forward, world_up), dim=0)
    up = torch.linalg.cross(right, forward)
    return right, up, forward


def _sample_volume(volume: torch.Tensor, points: torch.Tensor, *, bounds: float) -> torch.Tensor:
    """Trilinear logits at ``points`` ``[B, N, 3]`` from the ``[B, X, Y, Z]`` grid; points
    outside the box read as deep outside."""

    # grid_sample indexes its last three input axes as (D, H, W) with coordinates
    # ordered (x -> W, y -> H, z -> D), so the (x, y, z) grid becomes [B, 1, Z, Y, X].
    sampled = F.grid_sample(
        volume.permute(0, 3, 2, 1).unsqueeze(1),
        (points / bounds).view(points.shape[0], 1, 1, -1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(points.shape[0], -1)
    outside = (points.abs() > bounds).any(dim=-1)
    return sampled.masked_fill(outside, -1e4)


def render_occupancy_views(
    grid_logits: torch.Tensor,
    *,
    num_views: int,
    height: int,
    width: int,
    bounds: float = BOUNDS,
    elevation_deg: float = 20.0,
    samples_per_ray: int | None = None,
) -> torch.Tensor:
    """Shaded orthographic renders of an occupancy grid, ``[B, V, 3, H, W]`` in ``[0, 1]``.

    Views circle the shape at ``elevation_deg`` starting from the front (the
    conditioning image's side). Each pixel's ray is marched through the box; the
    first sample above ``SURFACE_LEVEL`` is the hit, the finite-difference normal
    there is lit by a headlight (ambient 0.25), and misses are white.
    """

    batch, side = grid_logits.shape[0], grid_logits.shape[1]
    device = grid_logits.device
    volume = grid_logits.float()
    steps = samples_per_ray or 2 * side
    diagonal = bounds * math.sqrt(3.0)
    depth = torch.linspace(-diagonal, diagonal, steps, device=device)
    ys = torch.linspace(bounds, -bounds, height, device=device)
    xs = torch.linspace(-bounds, bounds, width, device=device)
    pixel_y, pixel_x = torch.meshgrid(ys, xs, indexing="ij")
    eps = 2.0 * bounds / (side - 1)
    frames = []
    for view in range(num_views):
        right, up, forward = _view_basis(
            2.0 * math.pi * view / num_views, math.radians(elevation_deg), device
        )
        origins = pixel_x[..., None] * right + pixel_y[..., None] * up  # [H, W, 3]
        points = origins[None, :, :, None, :] + depth[:, None] * forward  # [1, H, W, S, 3]
        points = points.expand(batch, -1, -1, -1, -1).reshape(batch, -1, 3)
        inside = _sample_volume(volume, points, bounds=bounds).view(batch, height, width, steps)
        inside = inside > SURFACE_LEVEL
        hit = inside.any(dim=-1)
        first = inside.float().argmax(dim=-1)  # first True along the ray
        hit_points = origins[None] + depth[first][..., None] * forward  # [B, H, W, 3]
        flat = hit_points.reshape(batch, -1, 3)
        gradient = torch.stack(
            [
                _sample_volume(volume, flat + eps * axis, bounds=bounds)
                - _sample_volume(volume, flat - eps * axis, bounds=bounds)
                for axis in torch.eye(3, device=device)
            ],
            dim=-1,
        )
        # Logits grow inward, so the outward normal points down the gradient.
        normal = F.normalize(-gradient, dim=-1).view(batch, height, width, 3)
        lambert = (normal * -forward).sum(dim=-1).clamp(min=0.0)
        shade = 0.25 + 0.75 * lambert
        image = torch.where(hit, shade, torch.ones_like(shade))
        frames.append(image.unsqueeze(1).expand(-1, 3, -1, -1))
    return torch.stack(frames, dim=1).clamp(0.0, 1.0)


__all__ = [
    "BOUNDS",
    "SURFACE_LEVEL",
    "dense_grid_points",
    "query_occupancy_grid",
    "render_occupancy_views",
]

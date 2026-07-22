"""The deployed generative prior for the wave field: a small UNet DDPM.

The field has 64 nodes and is folded to an 8x8 image so that the same ``UNet2DModel`` used for
the image-domain examples carries it (a 1D UNet of this width underfits the field). Two priors
are trained with an identical architecture and differ only in their training target: the
oracle on the true fields, the curated one on the legacy MAP archive.

``clip_sample=False`` is load bearing throughout. The diffusers default clamps samples to
``[-1, 1]``, which would destroy the standardized amplitudes; amplitude is instead handled by
the per-node :class:`~priorlaundermat.utils.Normalizer`, whose statistics travel in the
checkpoint so that samples come back in physical units.
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, UNet2DModel

from priorlaundermat.utils import Normalizer

NNODE, SIDE = 64, 8


def make_unet(base: int = 64) -> UNet2DModel:
    """The epsilon-prediction UNet over the 8x8 folded field."""
    return UNet2DModel(
        sample_size=SIDE, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(base, base * 2, base * 4), norm_num_groups=min(32, base),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"))


def load_prior(ckpt_path: str, *, device="cpu", dtype: torch.dtype = torch.float32
               ) -> tuple[UNet2DModel, Normalizer]:
    """Load a trained prior and the per-node normalizer it was trained under.

    Args:
        ckpt_path: absolute path to a ``wave_prior_{arm}_seed{k}.pth`` checkpoint.
        device, dtype: placement for the network and the normalizer statistics. Posterior
            sampling needs float64 (the guidance gradient goes through the float64 solver);
            unconditional sampling runs happily in float32.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet = make_unet(64)
    unet.load_state_dict(ck["model"])
    unet = unet.to(device=device, dtype=dtype).eval()
    norm = Normalizer.from_stats(ck["norm_mean"].to(device=device, dtype=dtype),
                                 ck["norm_std"].to(device=device, dtype=dtype))
    return unet, norm


@torch.no_grad()
def sample_prior_ddim(unet: UNet2DModel, norm: Normalizer, n: int, *,
                      n_steps: int = 50, device="cpu", seed: int = 0) -> torch.Tensor:
    """Unconditional DDIM draws (deterministic) returned as ``(n, 64)`` physical fields."""
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False)
    sched.set_timesteps(n_steps)
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = next(unet.parameters()).dtype
    x = torch.randn(n, 1, SIDE, SIDE, device=device, dtype=dtype, generator=g)
    for t in sched.timesteps:
        eps = unet(x, t.to(device).expand(n)).sample
        x = sched.step(eps, t, x).prev_sample
    return norm.unnormalize(x.reshape(n, NNODE))

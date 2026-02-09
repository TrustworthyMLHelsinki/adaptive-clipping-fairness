from __future__ import annotations
from typing import Callable, Optional
import logging
import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _mark_as_processed,
)

logger = logging.getLogger(__name__)


class AutoClipOptimizer(DPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = True,
        optim_args: dict | None = None,
    ):
        optim_args = optim_args or {}
        gamma = float(optim_args.get("stability_const", 0.01))
        max_grad_norm = 1.0

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            optim_args=optim_args,
        )
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

        self.sample_size = 0
        self.avg_norm_last_step = 0.0

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none)
        self.sample_size = 0
        self.avg_norm_last_step = 0.0

    @torch.no_grad()
    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        if per_sample_norms.numel() > 0:
            self.sample_size += per_sample_norms.numel()
            self.avg_norm_last_step = float(per_sample_norms.mean().item())

        eps = 1e-12
        denom = per_sample_norms + self.gamma.to(per_sample_norms.device)
        denom = torch.clamp(denom, min=eps)
        per_sample_clip_factor = 1.0 / denom

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad
            _mark_as_processed(p.grad_sample)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        return super().pre_step(closure)

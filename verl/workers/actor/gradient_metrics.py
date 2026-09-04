from __future__ import annotations

from collections.abc import Iterable

import torch


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads]))


def clip_grad_norm_with_metrics(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> tuple[float, float]:
    parameters = list(parameters)
    pre_clip = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm).detach().item())
    post_clip = float(grad_norm(parameters).item())
    return pre_clip, post_clip

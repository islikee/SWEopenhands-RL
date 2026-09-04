import pytest
import torch

from verl.workers.actor.gradient_metrics import clip_grad_norm_with_metrics


def test_gradient_metrics_measure_distinct_pre_and_post_clip_norms():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])

    pre_clip, post_clip = clip_grad_norm_with_metrics([parameter], max_norm=1.0)

    assert pre_clip == pytest.approx(5.0)
    assert post_clip == pytest.approx(1.0)

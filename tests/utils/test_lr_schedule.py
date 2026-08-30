import torch

from verl.utils.torch_functional import get_constant_schedule_with_warmup


def test_constant_schedule_with_zero_warmup_keeps_initial_lr():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)

    assert optimizer.param_groups[0]["lr"] == 1e-6
    scheduler.step()
    assert scheduler.get_last_lr() == [1e-6]

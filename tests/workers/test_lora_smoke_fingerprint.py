import torch

from verl.workers.lora_utils import (
    distributed_any,
    distributed_max_float,
    find_parameter_fingerprint,
    max_lora_b_grad_abs,
)


def test_smoke_lora_fingerprint_can_follow_nonzero_lora_b_gradient():
    model = torch.nn.Module()
    model.register_parameter(
        "first_lora_B",
        torch.nn.Parameter(torch.zeros(2, 2), requires_grad=True),
    )
    model.register_parameter(
        "second_lora_B",
        torch.nn.Parameter(torch.ones(2, 2), requires_grad=True),
    )
    model.first_lora_B.grad = torch.zeros(2, 2)
    model.second_lora_B.grad = torch.full((2, 2), 0.5)

    name, grad_max_abs = max_lora_b_grad_abs(model)
    fingerprint = find_parameter_fingerprint(model, name)

    assert name == "second_lora_B"
    assert grad_max_abs == 0.5
    assert fingerprint["abs_sum"] == 4.0


def test_smoke_distributed_reducers_are_identity_without_process_group():
    assert distributed_max_float(0.25) == 0.25
    assert distributed_any(True) is True
    assert distributed_any(False) is False

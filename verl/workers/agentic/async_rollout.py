# TODO(haoran): stuck in the loop
# TODO(haoran): time control; loss_mask
# TODO(haoran): check reason for loading weight
import logging
import os
from functools import partial
from json import JSONDecodeError

import torch
import sglang as sgl
import torch.distributed
from omegaconf import DictConfig
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import Tool
from torch.distributed import DeviceMesh
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.agentic.codeact import CodeActAgentGroup
from verl.workers.agentic.rollout_limits import resolve_sglang_token_pool_limits

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def _pre_process_inputs(pad_token_id, token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = token_ids[non_pad_index:].tolist()
    return token_ids


class AsyncRollout(BaseRollout):

    def __init__(self, model_path, config: DictConfig, device_mesh: DeviceMesh):
        super().__init__()
        torch.distributed.barrier()
        # print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.tp_rank = device_mesh.get_local_rank(1)
        self.device_mesh = device_mesh
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        visible_devices: list[str | None] = [None] * device_mesh.size(1)
        torch.distributed.all_gather_object(visible_devices, cuda_visible_device, group=device_mesh.get_group(1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        print(
            f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}"
        )
        self.total_len = config.prompt_length + config.response_length
        token_pool_limits = resolve_sglang_token_pool_limits(config)
        print(f"async rollout {config.gpu_memory_utilization=}")
        torch.distributed.barrier()
        # print(f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}")
        if self.tp_rank == 0:
            self.engine = sgl.Engine(
                model_path=model_path,
                port=40000,
                dtype=config.dtype,
                max_total_tokens=token_pool_limits.max_total_tokens,
                max_prefill_tokens=token_pool_limits.max_prefill_tokens,
                disable_cuda_graph=config.get("disable_cuda_graph", False),
                enable_memory_saver=config.enable_memory_saver,
                mem_fraction_static=config.gpu_memory_utilization,
                tp_size=device_mesh.size(1),
                log_level="INFO",
                # enable_metrics=True,
            )
            print(f"nodedup {torch.distributed.get_rank() = } releasing memory occupation")
            self.engine.release_memory_occupation()
            print(f"nodedup {torch.distributed.get_rank() = } engine initialized")
        else:
            self.engine = None
        self.engine: sgl.srt.entrypoints.engine.Engine | None
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        torch.distributed.barrier()
        self.config = config
        self.task_type = config.task_type
        # print("Config sampling params:", config.sampling_params)
        self.sampling_params = dict(config.sampling_params)
        self.sampling_params.update({
            "skip_special_tokens": False,
        })

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto | None:
        # print(f"nodedup in generate seq {torch.distributed.get_rank()=} {self.tp_rank=} {prompts.non_tensor_batch=}")
        logger.info(f"nodedup in generate seq {torch.distributed.get_rank()=} {self.tp_rank=}")
        if self.tp_rank != 0:
            return None
        
        sampling_params = self.sampling_params.copy()
        # print("kwargs: ", kwargs)
        sampling_params.update(kwargs)
        print("final sampling params:", sampling_params)
        device = torch.cuda.current_device()
        
        codeact_agent_group = CodeActAgentGroup(
            batch=prompts,
            num_trajectories=prompts.meta_info.get("n_trajectories", self.config.n_trajectories),
            infer_engine=self.engine,
            max_prompt_length=self.config.get(
                "agent_max_prompt_length",
                self.config.prompt_length,
            ),
            max_response_length=self.config.response_length,
            max_starting_message_length=self.config.max_starting_message_length,
            max_parallel_agents=self.config.max_parallel_agents // self.device_mesh.size(0),
            max_eval_parallel_agents=self.config.max_eval_parallel_agents // self.device_mesh.size(0),
            max_iterations = self.config.max_iterations,
            tokenizer=self.engine.tokenizer_manager.tokenizer,
            sampling_params=sampling_params,
            device=device,
            log_messages_dir=self.config.log_messages_dir,
            remove_think_tokens=self.config.remove_think_tokens,
            qwen3_enable_thinking=self.config.qwen3_enable_thinking,
        )

        results = codeact_agent_group.run()
        logger.info(f"nodedup finish generate seq {torch.distributed.get_rank()=} {self.tp_rank=}")
        return results

from dataclasses import dataclass
import torch
from torchtune.models.qwen2_5 import qwen2_5_7b_base, qwen2_5_14b_base, qwen2_5_32b_base
from torchtune.models.llama3_1 import llama3_1_70b
from torchtune.modules import TransformerDecoder
from typing import Any, Callable

from .recipe import ComponentConfig


@dataclass
class Model:
    base_model: str
    tune_model: Callable[[], TransformerDecoder]
    tune_model_type: str
    tune_max_batch_tokens: int
    tune_optimizer: ComponentConfig
    vllm_named_arguments: dict[str, Any]
    tune_fsdp_cpu_offload: bool = False


def qwen_7b() -> Model:
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tune_model=qwen2_5_7b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=49152,
        tune_optimizer=ComponentConfig(
            "torch.optim.AdamW",
            lr=2e-5,
            fused=True,
        ),
        vllm_named_arguments={},
    )


def qwen_14b() -> Model:
    assert torch.cuda.device_count() >= 2, "Qwen-14B requires at least 2 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tune_model=qwen2_5_14b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=24576,
        tune_optimizer=ComponentConfig(
            "torchao.prototype.low_bit_optim.AdamW8bit",
            lr=2e-5,
        ),
        vllm_named_arguments={},
    )


def qwen_32b() -> Model:
    assert torch.cuda.device_count() >= 4, "Qwen-32B requires at least 4 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tune_model=qwen2_5_32b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=65536,
        tune_optimizer=ComponentConfig(
            "torch.optim.AdamW",
            lr=2e-5,
            fused=True,
        ),
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
    )


def llama_70b() -> Model:
    assert torch.cuda.device_count() >= 8, "Llama-70B requires at least 8 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        tune_model=llama3_1_70b,
        tune_model_type="LLAMA3",
        tune_max_batch_tokens=65536,
        tune_optimizer=ComponentConfig(
            "torch.optim.AdamW",
            lr=2e-5,
            fused=True,
        ),
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
    )

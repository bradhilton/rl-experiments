from dataclasses import dataclass
import torch
from torchtune.models.qwen2_5 import qwen2_5_7b_base, qwen2_5_14b_base, qwen2_5_32b_base
from torchtune.models.llama3_1 import llama3_1_8b, llama3_1_70b
from torchtune.modules import TransformerDecoder
from typing import Any, Callable, Literal


Optimizer = Literal[
    "torch.optim.AdamW",
    "torchao.prototype.low_bit_optim.AdamW8bit",
    "bitsandbytes.optim.PagedAdamW8bit",
]


@dataclass
class Model:
    base_model: str
    tune_model: Callable[[], TransformerDecoder]
    tune_model_type: str
    tune_max_batch_tokens: int
    tune_optimizer: Optimizer
    vllm_named_arguments: dict[str, Any]
    tune_fsdp_cpu_offload: bool = False
    tune_num_output_chunks: int = 8


def qwen_7b() -> Model:
    assert torch.cuda.device_count() >= 1, "Qwen-7B requires at least 1 GPU"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tune_model=qwen2_5_7b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=32768,
        tune_optimizer="torch.optim.AdamW",
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
    )


def theta_8b() -> Model:
    assert torch.cuda.device_count() >= 1, "Llama-8B requires at least 1 GPU"
    return Model(
        base_model="NousResearch/Hermes-2-Theta-Llama-3-8B",
        tune_model=llama3_1_8b,
        tune_model_type="LLAMA3",
        tune_max_batch_tokens=32768,
        tune_optimizer="torch.optim.AdamW",
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
    )


def qwen_14b() -> Model:
    assert torch.cuda.device_count() >= 2, "Qwen-14B requires at least 2 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tune_model=qwen2_5_14b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=32768,
        tune_optimizer="torch.optim.AdamW",
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
        tune_num_output_chunks=2,
    )


def qwen_32b() -> Model:
    assert torch.cuda.device_count() >= 4, "Qwen-32B requires at least 4 GPUs"
    return Model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tune_model=qwen2_5_32b_base,
        tune_model_type="QWEN2",
        tune_max_batch_tokens=32768,
        tune_optimizer="torch.optim.AdamW",
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
        tune_optimizer="torch.optim.AdamW",
        vllm_named_arguments={},
        tune_fsdp_cpu_offload=True,
    )

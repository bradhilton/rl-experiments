from dataclasses import dataclass
import torch
from torchtune.models.qwen2_5 import qwen2_5_7b_base
from torchtune.modules import TransformerDecoder
from typing import Any, Callable

from .recipe import ComponentConfig


device_count = torch.cuda.device_count()


@dataclass
class Model:
    base_model: str
    tune_model: Callable[[], TransformerDecoder]
    tune_model_type: str
    tune_max_batch_tokens: int
    tune_optimizer: ComponentConfig
    vllm_named_arguments: dict[str, Any]


qwen_7b = Model(
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tune_model=qwen2_5_7b_base,
    tune_model_type="QWEN2",
    tune_max_batch_tokens={
        0: 16384,
        1: 32768,
        2: 49152,
        4: 49152,
        8: 49152,
    }[device_count],
    tune_optimizer=(
        ComponentConfig(
            "torch.optim.AdamW",
            lr=2e-5,
            fused=True,
        )
        if torch.cuda.device_count() > 1
        else ComponentConfig(
            "bitsandbytes.optim.PagedAdamW8bit",
            lr=2e-5,
        )
    ),
    vllm_named_arguments={
        "max_model_len": 131072,
    },
)

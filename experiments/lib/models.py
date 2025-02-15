from dataclasses import dataclass
from torchtune.modules import TransformerDecoder
from typing import Any, Callable


@dataclass
class Model:
    tune_model: Callable[[], TransformerDecoder]
    tune_model_type: str
    tune_max_batch_tokens: int
    vllm_named_arguments: dict[str, Any]

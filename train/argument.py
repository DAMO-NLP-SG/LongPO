from dataclasses import dataclass, field, asdict
import transformers
from typing import Dict, Optional, Sequence



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ref_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    rope_theta: int = field(default=1000000)
    dpo_beta: float = field(default=0.01)
    # use_flashattn: Optional[bool] = field(default=True)




@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="/mnt/workspace/tmp")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_ring_attention: bool = field(default=False)
    dpo_lambda: float = field(default=0.1)
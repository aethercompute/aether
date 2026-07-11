import torch

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Iterable


@dataclass
class PretrainedSourceRepoFiles:
    files: list[str]


@dataclass
class PretrainedSourceStateDict:
    config_json: str
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    init_seed: int = 0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "LoraConfig":
        return cls(**value)


class CausalLM(ABC):

    @staticmethod
    @abstractmethod
    def from_pretrained(
        source: PretrainedSourceRepoFiles | PretrainedSourceStateDict,
        device: torch.device,
        attn_implementation: str,
        dp: int = 1,
        tp: int = 1,
        param_dtype: torch.dtype = torch.bfloat16,
        reduce_dtype: torch.dtype = torch.float32,
        fsdp_modules: Optional[Iterable[str]] = None,
        lora_config: Optional[LoraConfig] = None,
        adapter_source: Optional[
            PretrainedSourceRepoFiles | PretrainedSourceStateDict
        ] = None,
    ):
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[list[list[int]]] = None,
        num_logits_to_keep: Optional[int] = None,
        loss_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def named_parameters(self) -> dict[str, torch.Tensor]:
        pass

    def named_state_parameters(self) -> dict[str, torch.Tensor]:
        return self.named_parameters()

    def named_trainable_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.named_parameters().items()
            if parameter.requires_grad
        }

    @abstractmethod
    def get_config(self) -> dict:
        pass

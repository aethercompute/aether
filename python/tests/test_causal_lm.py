"""Tests for the ``causal_lm`` abstract interface and its source dataclasses.

The module imports torch at the top level, so we install a minimal torch stub
to exercise the pure-Python parts (dataclass construction, ABC enforcement)
without a real PyTorch install or GPU.
"""

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch):
    """Provide a fake torch so ``import aether.models.causal_lm`` succeeds."""
    fake = types.ModuleType("torch")
    fake.device = type("device", (), {})
    fake.dtype = type("dtype", (), {})
    fake.Tensor = type("Tensor", (), {})
    fake.bfloat16 = "bfloat16"
    fake.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", fake)
    # Force a fresh import so the stub is picked up.
    for mod in list(sys.modules):
        if mod.startswith("aether.models"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    yield


def test_pretrained_source_repo_files_holds_file_list():
    from aether.models.causal_lm import PretrainedSourceRepoFiles

    src = PretrainedSourceRepoFiles(files=["a.json", "b.safetensors"])
    assert src.files == ["a.json", "b.safetensors"]


def test_pretrained_source_repo_files_empty_list_allowed():
    from aether.models.causal_lm import PretrainedSourceRepoFiles

    src = PretrainedSourceRepoFiles(files=[])
    assert src.files == []


def test_pretrained_source_state_dict_holds_config_and_state():
    from aether.models.causal_lm import PretrainedSourceStateDict

    state = {"layer.0.weight": object()}
    src = PretrainedSourceStateDict(config_json='{"hidden": 4}', state_dict=state)
    assert src.config_json == '{"hidden": 4}'
    assert src.state_dict is state


def test_causal_lm_is_abstract_and_cannot_be_instantiated():
    from aether.models.causal_lm import CausalLM

    # CausalLM declares abstractmethods; direct instantiation must fail.
    with pytest.raises(TypeError):
        CausalLM()  # type: ignore[abstract]


def test_causal_lm_subclass_must_implement_all_abstract_methods():
    from aether.models.causal_lm import CausalLM

    # An incomplete subclass remains abstract.
    class Incomplete(CausalLM):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_causal_lm_concrete_subclass_implements_interface():
    from aether.models.causal_lm import CausalLM

    class Complete(CausalLM):
        @staticmethod
        def from_pretrained(source, device, attn_implementation, dp=1, tp=1,
                            param_dtype=None, reduce_dtype=None, fsdp_modules=None):
            return Complete()

        def forward(self, input_ids, labels, position_ids=None,
                    sequence_lengths=None, num_logits_to_keep=None, loss_scale=None):
            return (input_ids, labels)

        def named_parameters(self):
            return {}

        def get_config(self):
            return {"ok": True}

    obj = Complete()
    assert obj.get_config() == {"ok": True}
    logits, labels = obj.forward("ids", "labels")
    assert logits == "ids" and labels == "labels"
    assert obj.named_parameters() == {}


def test_pretrained_source_dataclasses_are_distinct_types():
    from aether.models.causal_lm import (
        PretrainedSourceRepoFiles,
        PretrainedSourceStateDict,
    )

    assert PretrainedSourceRepoFiles is not PretrainedSourceStateDict
    assert PretrainedSourceRepoFiles.__name__ == "PretrainedSourceRepoFiles"
    assert PretrainedSourceStateDict.__name__ == "PretrainedSourceStateDict"


def test_lora_config_round_trips_as_serializable_dict():
    from aether.models.causal_lm import LoraConfig

    config = LoraConfig(rank=4, alpha=8.0, dropout=0.1, init_seed=42)
    serialized = config.to_dict()

    assert serialized == {
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.1,
        "init_seed": 42,
    }
    assert LoraConfig.from_dict(serialized) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"alpha": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_lora_config_rejects_invalid_values(kwargs):
    from aether.models.causal_lm import LoraConfig

    with pytest.raises(ValueError):
        LoraConfig(**kwargs)


def test_factory_rejects_lora_for_non_hf_architecture():
    import torch

    from aether.models import LoraConfig, PretrainedSourceRepoFiles, make_causal_lm

    with pytest.raises(ValueError, match="only for the HfAuto"):
        make_causal_lm(
            "Torchtitan",
            PretrainedSourceRepoFiles([]),
            torch.device(),
            "eager",
            lora_config=LoraConfig(),
        )


@pytest.mark.parametrize(
    ("architecture", "module_name", "class_name", "supports_lora"),
    [
        ("HfAuto", "hf_transformers", "HfTransformersAuto", True),
        ("Torchtitan", "ttitan", "TorchtitanAuto", False),
    ],
)
def test_factory_routes_and_forwards_arguments(
    monkeypatch, architecture, module_name, class_name, supports_lora
):
    import torch

    from aether.models import PretrainedSourceRepoFiles, make_causal_lm

    calls = []
    sentinel = object()
    backend_module = types.ModuleType(f"aether.models.{module_name}")
    backend = type(
        class_name,
        (),
        {
            "from_pretrained": staticmethod(
                lambda **kwargs: calls.append(kwargs) or sentinel
            )
        },
    )
    setattr(backend_module, class_name, backend)
    monkeypatch.setitem(sys.modules, f"aether.models.{module_name}", backend_module)
    source = PretrainedSourceRepoFiles(["config.json"])
    device = torch.device()

    result = make_causal_lm(
        architecture,
        source,
        device,
        "eager",
        dp=2,
        tp=3,
        override_max_position_embeddings=128,
        param_dtype="params",
        reduce_dtype="reduce",
        fsdp_modules=["layer"],
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0] == {
        "source": source,
        "device": device,
        "attn_implementation": "eager",
        "dp": 2,
        "tp": 3,
        "override_max_position_embeddings": 128,
        "param_dtype": "params",
        "reduce_dtype": "reduce",
        "fsdp_modules": ["layer"],
        **({"lora_config": None, "adapter_source": None} if supports_lora else {}),
    }


def test_factory_rejects_unknown_architecture():
    import torch

    from aether.models import PretrainedSourceRepoFiles, make_causal_lm

    with pytest.raises(ValueError, match="Unknown architecture Unsupported"):
        make_causal_lm(
            "Unsupported",
            PretrainedSourceRepoFiles([]),
            torch.device(),
            "eager",
        )

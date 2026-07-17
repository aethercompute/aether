import sys
import types

import peft
import pytest
import torch
import transformers


@pytest.fixture
def hf_module(monkeypatch):
    monkey_patch = types.ModuleType("liger_kernel.transformers.monkey_patch")
    monkey_patch._apply_liger_kernel_to_instance = lambda **kwargs: None
    monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN = {}
    monkeypatch.setitem(sys.modules, "liger_kernel", types.ModuleType("liger_kernel"))
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers", types.ModuleType("transformers")
    )
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers.monkey_patch", monkey_patch
    )

    from aether.models import hf_transformers

    return hf_transformers


def tiny_model():
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=16,
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    return transformers.GPT2LMHeadModel(config)


def test_attach_lora_is_scoped_deterministic_and_freezes_base(hf_module):
    from aether.models import LoraConfig

    torch.manual_seed(11)
    first_base = tiny_model()
    second_base = tiny_model()
    second_base.load_state_dict(first_base.state_dict())
    expected_next_random = torch.rand(1)

    torch.manual_seed(11)
    tiny_model()
    tiny_model()
    first = hf_module._attach_lora(
        first_base, LoraConfig(rank=2, alpha=4, init_seed=123), torch.device("cpu")
    )
    actual_next_random = torch.rand(1)
    second = hf_module._attach_lora(
        second_base, LoraConfig(rank=2, alpha=4, init_seed=123), torch.device("cpu")
    )

    assert torch.equal(actual_next_random, expected_next_random)
    first_trainable = {
        name: value for name, value in first.named_parameters() if value.requires_grad
    }
    second_trainable = {
        name: value for name, value in second.named_parameters() if value.requires_grad
    }
    assert first_trainable
    assert first_trainable.keys() == second_trainable.keys()
    assert all("lora_" in name for name in first_trainable)
    assert all(
        torch.equal(first_trainable[name], second_trainable[name])
        for name in first_trainable
    )


def test_attach_lora_uses_explicit_adapter_dtype(hf_module):
    from aether.models import LoraConfig

    model = hf_module._attach_lora(
        tiny_model(),
        LoraConfig(rank=2, alpha=4),
        torch.device("cpu"),
        adapter_dtype=torch.bfloat16,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.dtype == torch.bfloat16 for parameter in trainable)


def test_lora_state_views_and_exports(hf_module):
    from aether.models import LoraConfig

    config = LoraConfig(rank=2, alpha=4, init_seed=7)
    model = hf_module._attach_lora(tiny_model(), config, torch.device("cpu"))
    wrapped = hf_module.HfTransformersAuto(
        model, model.config, world_mesh=None, device=torch.device("cpu")
    )
    wrapped.lora_config = config

    trainable = wrapped.named_trainable_parameters()
    state_parameters = wrapped.named_state_parameters()
    adapter = wrapped.adapter_state_dict()
    state = wrapped.named_state()
    live_before_merge = {
        name: tensor.detach().clone() for name, tensor in wrapped.named_state().items()
    }
    merged = wrapped.merged_state_dict()

    assert trainable and all(parameter.requires_grad for parameter in trainable.values())
    assert set(trainable) < set(state_parameters)
    assert adapter and all("lora_" in name for name in adapter)
    assert any("lora_" in name for name in state)
    assert not any("lora_" in name for name in merged)
    assert any("lora_" in name for name in wrapped.named_state())
    assert live_before_merge.keys() == wrapped.named_state().keys()
    assert all(
        torch.equal(value, wrapped.named_state()[name])
        for name, value in live_before_merge.items()
    )


def test_adapter_state_can_be_restored(hf_module):
    from aether.models import LoraConfig, PretrainedSourceStateDict

    config = LoraConfig(rank=2, alpha=4, init_seed=7)
    first = hf_module._attach_lora(tiny_model(), config, torch.device("cpu"))
    adapter_state = peft.get_peft_model_state_dict(first)
    for tensor in adapter_state.values():
        tensor.fill_(0.25)

    restored = hf_module._attach_lora(
        tiny_model(),
        config,
        torch.device("cpu"),
        PretrainedSourceStateDict(config_json="{}", state_dict=adapter_state),
    )

    restored_state = peft.get_peft_model_state_dict(restored)
    assert restored_state.keys() == adapter_state.keys()
    assert all(torch.equal(restored_state[name], adapter_state[name]) for name in adapter_state)


def test_internal_trainable_state_can_be_restored_from_p2p(hf_module):
    from aether.models import LoraConfig, PretrainedSourceStateDict

    config = LoraConfig(rank=2, alpha=4, init_seed=7)
    first = hf_module._attach_lora(tiny_model(), config, torch.device("cpu"))
    internal_state = {
        name: tensor.detach().clone()
        for name, tensor in first.named_parameters()
        if tensor.requires_grad
    }
    for tensor in internal_state.values():
        tensor.fill_(0.5)

    restored = hf_module._attach_lora(
        tiny_model(),
        config,
        torch.device("cpu"),
        PretrainedSourceStateDict(config_json="{}", state_dict=internal_state),
    )
    restored_internal = {
        name: tensor.detach()
        for name, tensor in restored.named_parameters()
        if tensor.requires_grad
    }

    assert restored_internal.keys() == internal_state.keys()
    assert all(
        torch.equal(restored_internal[name], internal_state[name])
        for name in internal_state
    )


def test_lora_rejects_distributed_modes_before_loading(hf_module):
    from aether.models import LoraConfig, PretrainedSourceStateDict

    source = PretrainedSourceStateDict(config_json="{}", state_dict={})
    with pytest.raises(ValueError, match="requires dp=1 and tp=1"):
        hf_module.HfTransformersAuto.from_pretrained(
            source,
            torch.device("cpu"),
            "eager",
            dp=2,
            lora_config=LoraConfig(),
        )

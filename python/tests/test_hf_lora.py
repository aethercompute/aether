import copy
import sys
import types
from types import SimpleNamespace

import peft
import pytest
import torch
import transformers
from safetensors.torch import save_file


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


def tiny_llama_model():
    config = transformers.LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    return transformers.LlamaForCausalLM(config)


@pytest.fixture
def local_hf_model_pair(hf_module, monkeypatch, tmp_path):
    from aether.models import PretrainedSourceRepoFiles

    torch.manual_seed(17)
    direct = tiny_llama_model()
    direct.save_pretrained(tmp_path, safe_serialization=True)
    direct = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path, attn_implementation="eager"
    )
    monkeypatch.setattr(hf_module, "maybe_compile_loss_function", lambda model: None)
    wrapped = hf_module.HfTransformersAuto.from_pretrained(
        PretrainedSourceRepoFiles([str(path) for path in sorted(tmp_path.iterdir())]),
        torch.device("cpu"),
        "eager",
        param_dtype=torch.float32,
    )
    direct.eval()
    wrapped.model.eval()
    return direct, wrapped


def test_from_pretrained_loads_tiny_local_checkpoint(local_hf_model_pair):
    direct, wrapped = local_hf_model_pair

    assert wrapped.config.model_type == direct.config.model_type
    assert wrapped.config.num_hidden_layers == direct.config.num_hidden_layers
    assert wrapped.config.hidden_size == direct.config.hidden_size
    assert wrapped.config.vocab_size == direct.config.vocab_size
    assert wrapped.named_state().keys() == direct.state_dict().keys()
    assert all(
        torch.equal(wrapped.named_state()[name], tensor)
        for name, tensor in direct.state_dict().items()
    )


@pytest.mark.oracle
def test_forward_logits_match_direct_transformers_model(local_hf_model_pair):
    direct, wrapped = local_hf_model_pair
    input_ids = torch.tensor([[0, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.long)

    with torch.no_grad():
        expected = direct(input_ids, use_cache=False, return_dict=True).logits
        actual, loss = wrapped.forward(input_ids, labels=None)

    assert loss is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.oracle
def test_forward_loss_matches_direct_transformers_model(local_hf_model_pair):
    direct, wrapped = local_hf_model_pair
    input_ids = torch.tensor([[0, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.long)
    labels = torch.tensor([[0, 3, 5, 7], [2, 4, -100, 8]], dtype=torch.long)

    with torch.no_grad():
        expected = direct(
            input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        ).loss
        _, actual = wrapped.forward(input_ids, labels=labels)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_from_pretrained_honors_cpu_device_and_parameter_dtype(
    hf_module, monkeypatch, tmp_path
):
    from aether.models import PretrainedSourceRepoFiles

    torch.manual_seed(23)
    tiny_llama_model().save_pretrained(tmp_path, safe_serialization=True)
    monkeypatch.setattr(hf_module, "maybe_compile_loss_function", lambda model: None)
    wrapped = hf_module.HfTransformersAuto.from_pretrained(
        PretrainedSourceRepoFiles([str(path) for path in sorted(tmp_path.iterdir())]),
        torch.device("cpu"),
        "eager",
        param_dtype=torch.bfloat16,
    )

    parameters = list(wrapped.model.parameters())
    assert parameters
    assert all(parameter.device == torch.device("cpu") for parameter in parameters)
    assert all(parameter.dtype == torch.bfloat16 for parameter in parameters)


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


def test_merged_state_matches_direct_peft_merge(hf_module):
    from aether.models import LoraConfig

    config = LoraConfig(rank=2, alpha=4, init_seed=7)
    model = hf_module._attach_lora(tiny_model(), config, torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                parameter.fill_(0.25)
            elif "lora_B" in name:
                parameter.fill_(0.5)
    wrapped = hf_module.HfTransformersAuto(
        model, model.config, world_mesh=None, device=torch.device("cpu")
    )
    wrapped.lora_config = config

    actual = wrapped.merged_state_dict()
    expected = copy.deepcopy(model).merge_and_unload().state_dict()

    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], expected[name]) for name in expected)


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


def test_adapter_only_state_can_be_saved_and_reloaded_from_disk(hf_module, tmp_path):
    from aether.models import LoraConfig, PretrainedSourceRepoFiles

    config = LoraConfig(rank=2, alpha=4, init_seed=7)
    first = hf_module._attach_lora(tiny_model(), config, torch.device("cpu"))
    adapter_state = peft.get_peft_model_state_dict(first)
    for tensor in adapter_state.values():
        tensor.fill_(0.375)
    path = tmp_path / "adapter_model.safetensors"
    save_file(adapter_state, path)

    restored = hf_module._attach_lora(
        tiny_model(),
        config,
        torch.device("cpu"),
        PretrainedSourceRepoFiles([str(path)]),
    )
    restored_state = peft.get_peft_model_state_dict(restored)

    assert adapter_state and all("lora_" in name for name in adapter_state)
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


def test_checkpoint_key_validation_handles_missing_unexpected_and_tied_keys(hf_module):
    untied = SimpleNamespace(tie_word_embeddings=False)
    tied = SimpleNamespace(tie_word_embeddings=True)
    expected = {"transformer.weight", "lm_head.weight"}

    with pytest.raises(RuntimeError, match="Missing parameter.*transformer.weight"):
        hf_module._validate_checkpoint_keys(expected, {"lm_head.weight"}, untied)
    with pytest.raises(RuntimeError, match="Unexpected parameter.*extra.weight"):
        hf_module._validate_checkpoint_keys(
            expected,
            {"transformer.weight", "lm_head.weight", "extra.weight"},
            untied,
        )

    hf_module._validate_checkpoint_keys(expected, {"transformer.weight"}, tied)
    hf_module._validate_checkpoint_keys(
        expected,
        {"transformer.weight", "lm_head.weight"},
        tied,
    )

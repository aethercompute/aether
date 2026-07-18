import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def ttitan_module(monkeypatch):
    argument_classes = {
        name: type(name, (_Args,), {})
        for name in (
            "DeepSeekV3ModelArgs",
            "GptOssModelArgs",
            "MoEArgs",
            "Qwen3ModelArgs",
            "Qwen3NextModelArgs",
            "RoPEScalingArgs",
            "TransformerModelArgs",
        )
    }
    modules = {
        "torchtitan": _module("torchtitan"),
        "torchtitan.config": _module("torchtitan.config", JobConfig=object),
        "torchtitan.config.job_config": _module(
            "torchtitan.config.job_config", PEFT=object
        ),
        "torchtitan.components": _module("torchtitan.components"),
        "torchtitan.components.loss": _module(
            "torchtitan.components.loss", build_cross_entropy_loss=lambda config: None
        ),
        "torchtitan.distributed": _module(
            "torchtitan.distributed", ParallelDims=object
        ),
        "torchtitan.distributed.utils": _module(
            "torchtitan.distributed.utils", maybe_enable_amp=lambda **kwargs: nullcontext()
        ),
        "torchtitan.experiments": _module("torchtitan.experiments"),
        "torchtitan.experiments.gpt_oss": _module(
            "torchtitan.experiments.gpt_oss", get_train_spec=lambda: None
        ),
        "torchtitan.experiments.gpt_oss.model": _module(
            "torchtitan.experiments.gpt_oss.model"
        ),
        "torchtitan.experiments.gpt_oss.model.args": _module(
            "torchtitan.experiments.gpt_oss.model.args",
            GptOssModelArgs=argument_classes["GptOssModelArgs"],
        ),
        "torchtitan.experiments.qwen3_next": _module(
            "torchtitan.experiments.qwen3_next", get_train_spec=lambda: None
        ),
        "torchtitan.experiments.qwen3_next.model": _module(
            "torchtitan.experiments.qwen3_next.model"
        ),
        "torchtitan.experiments.qwen3_next.model.args": _module(
            "torchtitan.experiments.qwen3_next.model.args",
            Qwen3NextModelArgs=argument_classes["Qwen3NextModelArgs"],
        ),
        "torchtitan.models": _module("torchtitan.models"),
        "torchtitan.models.deepseek_v3": _module(
            "torchtitan.models.deepseek_v3", get_train_spec=lambda: None
        ),
        "torchtitan.models.deepseek_v3.model": _module(
            "torchtitan.models.deepseek_v3.model"
        ),
        "torchtitan.models.deepseek_v3.model.args": _module(
            "torchtitan.models.deepseek_v3.model.args",
            DeepSeekV3ModelArgs=argument_classes["DeepSeekV3ModelArgs"],
        ),
        "torchtitan.models.llama3": _module(
            "torchtitan.models.llama3", get_train_spec=lambda: None
        ),
        "torchtitan.models.llama3.model": _module("torchtitan.models.llama3.model"),
        "torchtitan.models.llama3.model.args": _module(
            "torchtitan.models.llama3.model.args",
            TransformerModelArgs=argument_classes["TransformerModelArgs"],
            RoPEScalingArgs=argument_classes["RoPEScalingArgs"],
        ),
        "torchtitan.models.moe": _module(
            "torchtitan.models.moe", MoEArgs=argument_classes["MoEArgs"]
        ),
        "torchtitan.models.qwen3": _module(
            "torchtitan.models.qwen3", get_train_spec=lambda: None
        ),
        "torchtitan.models.qwen3.model": _module("torchtitan.models.qwen3.model"),
        "torchtitan.models.qwen3.model.args": _module(
            "torchtitan.models.qwen3.model.args",
            Qwen3ModelArgs=argument_classes["Qwen3ModelArgs"],
        ),
        "torchtitan.tools": _module("torchtitan.tools"),
        "torchtitan.tools.utils": _module(
            "torchtitan.tools.utils",
            get_device_info=lambda: ("cpu", None),
            set_default_dtype=lambda dtype: nullcontext(),
        ),
    }
    liger_patch = _module(
        "liger_kernel.transformers.monkey_patch",
        _apply_liger_kernel_to_instance=lambda **kwargs: None,
        MODEL_TYPE_TO_APPLY_LIGER_FN={},
    )
    modules.update(
        {
            "liger_kernel": _module("liger_kernel"),
            "liger_kernel.transformers": _module("liger_kernel.transformers"),
            "liger_kernel.transformers.monkey_patch": liger_patch,
        }
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "aether.models.ttitan", raising=False)
    return importlib.import_module("aether.models.ttitan")


def _config(model_type):
    return SimpleNamespace(
        model_type=model_type,
        max_position_embeddings=128,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rope_scaling={
            "factor": 2.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
        },
        head_dim=4,
        intermediate_size=48,
        attention_bias=True,
        moe_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.5,
        swiglu_limit=7.0,
        sliding_window=16,
        num_local_experts=4,
        hidden_act="silu",
        partial_rotary_factor=0.5,
        decoder_sparse_step=2,
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
    )


@pytest.mark.parametrize(
    ("model_type", "class_name", "expected"),
    [
        ("llama", "TransformerModelArgs", {"use_qkv_bias": False}),
        ("qwen2", "TransformerModelArgs", {"use_qkv_bias": True}),
        ("seed_oss", "TransformerModelArgs", {"use_qkv_bias": True}),
        ("qwen3_moe", "Qwen3ModelArgs", {"moe_enabled": True}),
        ("deepseek_v3", "DeepSeekV3ModelArgs", {"n_dense_layers": 1}),
        ("gpt_oss", "GptOssModelArgs", {"sliding_window_size": 16}),
        ("qwen3_next", "Qwen3NextModelArgs", {"full_attention_interval": 4}),
    ],
)
def test_convert_config_for_every_supported_family(
    ttitan_module, model_type, class_name, expected
):
    converted = ttitan_module.TorchtitanAuto.convert_config(_config(model_type))

    assert type(converted).__name__ == class_name
    assert converted.max_seq_len == 128
    assert converted.vocab_size == 32
    for name, value in expected.items():
        assert getattr(converted, name) == value
    if hasattr(converted, "moe_args"):
        assert converted.moe_args.num_experts == 4
        assert converted.moe_args.top_k == 2


def test_convert_config_rejects_unknown_architecture(ttitan_module):
    with pytest.raises(ValueError, match="Unsupported model_type `unknown`"):
        ttitan_module.TorchtitanAuto.convert_config(_config("unknown"))


@pytest.mark.parametrize("missing", ["max_position_embeddings", "hidden_size"])
def test_convert_config_reports_missing_required_fields(ttitan_module, missing):
    config = _config("llama")
    delattr(config, missing)

    if missing == "max_position_embeddings":
        with pytest.raises(ValueError, match="max sequence length"):
            ttitan_module.TorchtitanAuto.convert_config(config)
    else:
        with pytest.raises(AttributeError, match=missing):
            ttitan_module.TorchtitanAuto.convert_config(config)


def test_state_key_prefixes_are_normalized_for_load_and_parameter_views(
    ttitan_module,
):
    prefix = ttitan_module._CHECKPOINT_PREFIX + ttitan_module.COMPILE_PREFIX
    destination = torch.zeros(2)

    class WrappedModel:
        def state_dict(self):
            return {f"{prefix}weight": destination}

        def named_parameters(self):
            return [(f"{prefix}weight", destination)]

    model = WrappedModel()
    ttitan_module.TorchtitanAuto._load_into_model(
        model, {"weight": torch.tensor([1.0, 2.0])}
    )
    wrapped = ttitan_module.TorchtitanAuto(
        model,
        None,
        None,
        None,
        None,
        torch.device("cpu"),
        nullcontext(),
        SimpleNamespace(world_mesh=None),
    )

    assert torch.equal(destination, torch.tensor([1.0, 2.0]))
    assert wrapped.named_parameters().keys() == {"weight"}
    assert wrapped.named_parameters()["weight"] is destination


@pytest.mark.oracle
def test_tiny_torchtitan_wrapper_forward_and_loss_on_cpu(ttitan_module):
    torch.manual_seed(31)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.output = torch.nn.Linear(4, 8, bias=False)

        def forward(self, tokens, position_ids=None):
            return self.output(self.embedding(tokens))

    model = TinyModel()
    loss_fn = lambda logits, labels: F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    wrapped = ttitan_module.TorchtitanAuto(
        model,
        loss_fn,
        None,
        None,
        None,
        torch.device("cpu"),
        nullcontext(),
        SimpleNamespace(world_mesh=None),
    )
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    labels = torch.tensor([[0, 1, 2, 3], [4, 5, -100, 7]])

    expected_logits = model(tokens=tokens)
    expected_loss = loss_fn(expected_logits[:, :-1, :], labels[:, 1:])
    logits, loss = wrapped.forward(tokens, labels)

    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(loss, expected_loss)

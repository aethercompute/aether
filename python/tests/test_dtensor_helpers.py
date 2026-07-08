import importlib.util
import sys
import types
from pathlib import Path


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.grad = None

    def copy_(self, other):
        self.value = other.value
        return self

    def zero_(self):
        self.value = 0
        return self


class FakeDTensor:
    def __init__(self, local, *, device_mesh="mesh", placements=("shard",)):
        self._local = local
        self.device_mesh = device_mesh
        self.placements = placements
        self.shape = (1,)
        self.dtype = "dtype"
        self.layout = "layout"
        self.grad = None

    def to_local(self):
        return self._local

    def full_tensor(self):
        return self._local


def load_dtensor_helpers(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor

    fake_distributed = types.ModuleType("torch.distributed")
    fake_tensor_mod = types.ModuleType("torch.distributed.tensor")

    def distribute_tensor(tensor, *, device_mesh, placements):
        return FakeDTensor(tensor, device_mesh=device_mesh, placements=placements)

    def zeros(shape, *, dtype, layout, device_mesh, placements):
        result = FakeDTensor(FakeTensor(0), device_mesh=device_mesh, placements=placements)
        result.shape = shape
        result.dtype = dtype
        result.layout = layout
        return result

    fake_tensor_mod.DTensor = FakeDTensor
    fake_tensor_mod.distribute_tensor = distribute_tensor
    fake_tensor_mod.zeros = zeros
    fake_distributed.tensor = fake_tensor_mod
    fake_torch.distributed = fake_distributed

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_distributed)
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", fake_tensor_mod)

    module_path = Path(__file__).parents[1] / "python" / "aether" / "dtensor_helpers.py"
    spec = importlib.util.spec_from_file_location("dtensor_helpers_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_grad_uses_public_to_local_for_dtensor(monkeypatch):
    helpers = load_dtensor_helpers(monkeypatch)
    tensor = FakeDTensor(FakeTensor(1))
    tensor.grad = FakeDTensor(FakeTensor(2))

    helpers.set_grad(tensor, FakeTensor(9))

    assert tensor.grad.to_local().value == 9


def test_zero_grad_uses_public_to_local_for_dtensor(monkeypatch):
    helpers = load_dtensor_helpers(monkeypatch)
    tensor = FakeDTensor(FakeTensor(1))
    tensor.grad = FakeDTensor(FakeTensor(9))

    helpers.zero_grad(tensor)

    assert tensor.grad.to_local().value == 0


def test_set_and_zero_grad_for_plain_tensor(monkeypatch):
    helpers = load_dtensor_helpers(monkeypatch)
    tensor = FakeTensor(1)
    tensor.grad = FakeTensor(2)

    helpers.set_grad(tensor, FakeTensor(7))
    assert tensor.grad.value == 7

    helpers.zero_grad(tensor)
    assert tensor.grad.value == 0

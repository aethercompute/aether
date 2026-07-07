"""Tests for the sidecar operation dataclasses.

These dataclasses define the wire protocol between the Rust trainer and the
Python sidecar process. They carry no torch dependency, so they can be unit
tested directly without a GPU or a distributed backend.
"""

import pytest

from aether.sidecar import api


def test_operation_base_carries_kind():
    op = api.Operation(operation="hyperparameters")
    assert op.operation == "hyperparameters"


def test_hyperparameters_inherits_operation():
    hp = api.Hyperparameters(
        operation="hyperparameters",
        lr_scheduler={"base_lr": 0.1},
        optimizer={"type": "adamw"},
        micro_batch_size=4,
        grad_accum_in_fp32=False,
    )
    assert hp.operation == "hyperparameters"
    assert hp.micro_batch_size == 4
    assert hp.grad_accum_in_fp32 is False
    assert hp.lr_scheduler == {"base_lr": 0.1}


def test_hyperparameters_accepts_fp32_grad_accum():
    hp = api.Hyperparameters(
        operation="hyperparameters",
        lr_scheduler={},
        optimizer={},
        micro_batch_size=1,
        grad_accum_in_fp32=True,
    )
    assert hp.grad_accum_in_fp32 is True


def test_distro_results_metadata_round_trips_fields():
    meta = api.DistroResultsMetadata(
        sparse_idx_size=[[10], [20]],
        sparse_idx_dtype=4,  # int64
        sparse_val_size=[[10], [20]],
        sparse_val_dtype=15,  # bfloat16
        xshape=[[4, 5], [6, 7]],
        totalk=[100, 200],
    )
    assert meta.sparse_idx_size == [[10], [20]]
    assert meta.sparse_val_dtype == 15
    assert meta.xshape == [[4, 5], [6, 7]]
    assert meta.totalk == [100, 200]
    # the invariants the receiver enforces (parallel lists of equal length).
    assert len(meta.sparse_idx_size) == len(meta.sparse_val_size)
    assert len(meta.sparse_val_size) == len(meta.xshape)
    assert len(meta.xshape) == len(meta.totalk)


def test_train_operation_required_fields():
    op = api.TrainOperation(
        operation="train",
        step=3,
        zero_optim=False,
        results_len=0,
        batch_id=(0, 10),
        batch_shape=[2, 16],
        batch_has_labels=True,
        batch_has_position_ids=False,
    )
    assert op.step == 3
    assert op.batch_id == (0, 10)
    assert op.batch_shape == [2, 16]
    assert op.batch_has_labels is True
    assert op.batch_has_position_ids is False
    # Optional fields default to None.
    assert op.warmup_lr_between is None
    assert op.results_metadata is None
    assert op.batch_sequence_lengths is None


def test_train_operation_with_optionals():
    op = api.TrainOperation(
        operation="train",
        step=1,
        zero_optim=True,
        results_len=2,
        batch_id=(5, 9),
        batch_shape=[1, 8],
        batch_has_labels=False,
        batch_has_position_ids=True,
        batch_sequence_lengths=[[8]],
        warmup_lr_between=(0, 100),
        results_metadata=api.DistroResultsMetadata(
            sparse_idx_size=[[1]],
            sparse_idx_dtype=4,
            sparse_val_size=[[1]],
            sparse_val_dtype=6,
            xshape=[[1]],
            totalk=[1],
        ),
    )
    assert op.zero_optim is True
    assert op.warmup_lr_between == (0, 100)
    assert op.results_metadata is not None
    assert op.batch_sequence_lengths == [[8]]


def test_optimize_operation_defaults():
    op = api.OptimizeOperation(operation="optimize", step=7, results_len=0)
    assert op.step == 7
    assert op.warmup_lr_between is None
    assert op.results_metadata is None


def test_forward_operation_defaults():
    op = api.ForwardOperation(
        operation="forward",
        batch_shape=[1, 4],
        batch_has_labels=False,
        batch_has_position_ids=False,
    )
    assert op.batch_shape == [1, 4]
    assert op.num_logits_to_keep is None
    assert op.loss_scale is None
    assert op.batch_sequence_lengths is None


def test_forward_operation_with_optionals():
    op = api.ForwardOperation(
        operation="forward",
        batch_shape=[2, 8],
        batch_has_labels=True,
        batch_has_position_ids=True,
        batch_sequence_lengths=[[8], [8]],
        num_logits_to_keep=1,
        loss_scale=0.5,
    )
    assert op.num_logits_to_keep == 1
    assert op.loss_scale == 0.5
    assert op.batch_sequence_lengths == [[8], [8]]


def test_train_operation_missing_required_field_raises():
    # dataclass __init__ requires every field without a default; omitting one
    # must raise a TypeError rather than silently producing an invalid object.
    with pytest.raises(TypeError):
        api.TrainOperation(
            operation="train",
            step=1,
            zero_optim=False,
            results_len=0,
            batch_id=(0, 1),
            # batch_shape omitted on purpose
            batch_has_labels=True,
            batch_has_position_ids=False,
        )


@pytest.mark.parametrize(
    "kind",
    ["hyperparameters", "train", "optimize", "extract", "truncate_bf16", "forward", "exit"],
)
def test_operation_kinds_are_strings(kind):
    # Every operation kind the sidecar dispatches on must round-trip through
    # the base Operation dataclass.
    assert api.Operation(operation=kind).operation == kind

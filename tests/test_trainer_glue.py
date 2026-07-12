"""Regression tests for src/trainer/optimizers.py::get_optimizer and
src/trainer/scheduler.py::get_scheduler.

These functions didn't exist at all until this test suite was added -
train_heron.py imported them but only unused optimizer/scheduler
*classes* were present, so any training run crashed on import.
"""
import types

import pytest
import torch
import torch.nn as nn

from src.trainer.optimizers import get_optimizer
from src.trainer.scheduler import get_scheduler


def make_args(**overrides):
    defaults = dict(
        optim='adamw', lr_init=1e-3, lr_final=1e-5, weight_decay=1e-4,
        lr_decay=-1, lr_decay_type='cos', lr_minibatch=True, num_epoch=5,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@pytest.fixture
def model():
    return nn.Linear(4, 1)


@pytest.mark.parametrize('optim_name', ['adamw', 'adam', 'sgd'])
def test_get_optimizer_builds_expected_type(model, optim_name):
    args = make_args(optim=optim_name)
    optimizer = get_optimizer(args, model)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(args.lr_init)


def test_get_optimizer_rejects_unknown_optim(model):
    args = make_args(optim='not-a-real-optimizer')
    with pytest.raises(ValueError):
        get_optimizer(args, model)


@pytest.mark.parametrize('decay_type', ['cos', 'lin', 'exp'])
@pytest.mark.parametrize('lr_minibatch', [True, False])
def test_get_scheduler_steps_without_error(model, decay_type, lr_minibatch):
    args = make_args(lr_decay_type=decay_type, lr_minibatch=lr_minibatch)
    optimizer = get_optimizer(args, model)
    train_loader = [None] * 10  # only len() is used
    scheduler = get_scheduler(args, optimizer, train_loader)

    for _ in range(3):
        optimizer.step()
        scheduler.step()


def test_get_scheduler_rejects_unknown_decay_type(model):
    args = make_args(lr_decay_type='not-a-real-schedule')
    optimizer = get_optimizer(args, model)
    with pytest.raises(ValueError):
        get_scheduler(args, optimizer, [None] * 10)

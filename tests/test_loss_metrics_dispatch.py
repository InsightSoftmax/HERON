"""LOSS_FUNCTIONS / METRICS_FUNCTIONS dispatch by output_head.

Trainer used to hardcode sharpe_loss/compute_all_metrics regardless of
output_head. These dicts are the single place a portfolio-compatible loss
and metrics function need to be dropped in once their semantics are decided
(see portfolio_loss / compute_portfolio_metrics docstrings).
"""
import pytest

from src.trainer.trainer import LOSS_FUNCTIONS, sharpe_loss, portfolio_loss
from src.models.metrics import METRICS_FUNCTIONS, compute_all_metrics, compute_portfolio_metrics


def test_loss_dispatch_has_entry_per_output_head():
    assert LOSS_FUNCTIONS['crosssectional'] is sharpe_loss
    assert LOSS_FUNCTIONS['portfolio'] is portfolio_loss


def test_metrics_dispatch_has_entry_per_output_head():
    assert METRICS_FUNCTIONS['crosssectional'] is compute_all_metrics
    assert METRICS_FUNCTIONS['portfolio'] is compute_portfolio_metrics


def test_portfolio_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        portfolio_loss(None, None)
    with pytest.raises(NotImplementedError):
        compute_portfolio_metrics(None, None)

"""
metrics.py — HERON financial evaluation metrics

Replaces metrics_classifier.py / metrics_cov.py from PELICAN.

PELICAN metrics: accuracy, AUC, background rejection (jet physics classification).
HERON metrics: Information Coefficient (IC), Sharpe ratio, max drawdown, hit rate.

These are the standard metrics used in systematic/quantitative equity strategies.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def information_coefficient(predictions, targets, mask=None):
    """
    Rank Information Coefficient: Spearman correlation between predicted
    scores and realised cross-sectional returns.

    IC is the primary alpha quality metric in quant research. An IC of 0.05
    with t-stat > 2.0 is generally considered tradeable.

    Parameters
    ----------
    predictions : Tensor [B, N]  per-asset predicted scores
    targets     : Tensor [B, N]  per-asset realised returns
    mask        : BoolTensor [B, N] or None

    Returns
    -------
    ic_mean : float   mean IC across cross-sections
    ic_std  : float   std of IC across cross-sections
    ic_tstat: float   t-statistic of IC (ic_mean / ic_std * sqrt(B))
    """
    B = predictions.shape[0]
    ics = []

    for b in range(B):
        pred_b = predictions[b]
        tgt_b = targets[b]
        if mask is not None:
            m = mask[b]
            pred_b = pred_b[m]
            tgt_b = tgt_b[m]

        if len(pred_b) < 3:
            continue

        # Rank both series
        pred_rank = _rank(pred_b)
        tgt_rank = _rank(tgt_b)

        # Pearson correlation of ranks = Spearman correlation
        ic = _pearson(pred_rank, tgt_rank).item()
        ics.append(ic)

    if len(ics) == 0:
        return 0.0, 1.0, 0.0

    ics = np.array(ics)
    ic_mean = ics.mean()
    ic_std = ics.std() + 1e-8
    ic_tstat = ic_mean / ic_std * np.sqrt(len(ics))
    return ic_mean, ic_std, ic_tstat


def portfolio_sharpe(predictions, targets, mask=None, annualise=True,
                     rebalance_freq=21, long_short=True):
    """
    Out-of-sample Sharpe ratio of a long-short portfolio constructed from
    the model's cross-sectional alpha scores.

    Portfolio construction: rank assets by predicted score, go long top quintile
    and short bottom quintile, equal-weighted within each leg.

    Parameters
    ----------
    predictions : Tensor [B, N]  per-asset predicted alpha scores
    targets     : Tensor [B, N]  per-asset realised forward returns
    mask        : BoolTensor [B, N] or None
    annualise   : bool   annualise by sqrt(252/rebalance_freq)
    rebalance_freq : int  rebalancing frequency in calendar days

    Returns
    -------
    sharpe : float
    portfolio_returns : np.ndarray [B]
    """
    B = predictions.shape[0]
    port_returns = []

    for b in range(B):
        pred_b = predictions[b]
        tgt_b = targets[b]
        if mask is not None:
            m = mask[b]
            pred_b = pred_b[m]
            tgt_b = tgt_b[m]

        N = len(pred_b)
        if N < 10:
            continue

        # Rank-based long/short portfolio
        ranks = _rank(pred_b)  # 0 = lowest predicted, N-1 = highest
        quintile = N // 5
        long_mask = ranks >= (N - quintile)
        short_mask = ranks < quintile

        if long_short:
            long_ret = tgt_b[long_mask].mean().item()
            short_ret = tgt_b[short_mask].mean().item()
            port_ret = long_ret - short_ret
        else:
            port_ret = tgt_b[long_mask].mean().item()

        port_returns.append(port_ret)

    if len(port_returns) < 2:
        return 0.0, np.array(port_returns)

    port_returns = np.array(port_returns)
    mean_ret = port_returns.mean()
    std_ret = port_returns.std() + 1e-8
    sharpe = mean_ret / std_ret
    if annualise:
        sharpe *= np.sqrt(252 / rebalance_freq)
    return sharpe, port_returns


def max_drawdown(portfolio_returns):
    """Maximum drawdown of a return series."""
    cum = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / (running_max + 1e-8)
    return drawdown.min()


def compute_all_metrics(predictions, targets, mask=None, prefix=''):
    """
    Compute and log all standard HERON metrics.

    Returns a dict suitable for logging / CSV output.
    """
    preds_np = predictions.detach().cpu()
    tgts_np = targets.detach().cpu()
    mask_np = mask.cpu() if mask is not None else None

    ic_mean, ic_std, ic_tstat = information_coefficient(preds_np, tgts_np, mask_np)
    sharpe, port_rets = portfolio_sharpe(preds_np, tgts_np, mask_np)
    mdd = max_drawdown(port_rets) if len(port_rets) > 1 else 0.0

    metrics = {
        f'{prefix}IC_mean': ic_mean,
        f'{prefix}IC_std': ic_std,
        f'{prefix}IC_tstat': ic_tstat,
        f'{prefix}Sharpe': sharpe,
        f'{prefix}MaxDrawdown': mdd,
        f'{prefix}nobs': len(port_rets),
    }

    logger.info(
        f'{prefix}IC={ic_mean:.4f} (t={ic_tstat:.2f})  '
        f'Sharpe={sharpe:.3f}  MaxDD={mdd:.1%}'
    )
    return metrics


# --- Helpers ---

def _rank(x):
    """Convert tensor to float ranks (0-indexed)."""
    idx = x.argsort()
    ranks = torch.zeros_like(idx, dtype=x.dtype)
    ranks[idx] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return ranks


def _pearson(x, y):
    """Pearson correlation coefficient between two 1D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()) + 1e-8
    return (x * y).sum() / denom

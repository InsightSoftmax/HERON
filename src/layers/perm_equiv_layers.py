# Based on https://github.com/horacepan/permeqlayers/blob/main/equivariant_layers.py
# Adapted for HERON (Hierarchical Equivariant Return Optimization Network)
#
# These are the core permutation-equivariant operations. They operate on pairwise
# financial signals between assets — correlations, factor cross-exposures, and
# learned bilinear feature interactions. All 15 basis elements of Eq₂→₂ are
# available, since no symmetry constraint beyond permutation applies to this
# input space, giving HERON substantial architectural flexibility.

import torch
import torch.nn as nn


def check_shape(x, shape):
    assert len(x.shape) == len(shape)
    for xs, s in zip(x.shape, shape):
        assert xs == s


# --- Masked aggregation helpers ---
# "Masked" means we only aggregate over real assets, ignoring zero-padded slots
# in batches where different cross-sections have different numbers of assets.

def masked_mean(x, nobj, dim=None, keepdims=False):
    x = torch.sum(x, dim=dim, keepdims=keepdims)
    if type(dim) != int:
        nobj = nobj ** (len(dim))
    nobj = nobj.view([-1] + [1, ] * (len(x.shape) - 1))
    x = x / nobj
    return x


def masked_amax(x, nobj, dim=None, keepdims=False):
    x = torch.amax(x, dim=dim, keepdims=keepdims)
    x = x - nobj.log().view([-1] + [1, ] * (len(x.shape) - 1))
    return x


def masked_amin(x, nobj, dim=None, keepdims=False):
    x = torch.amin(x, dim=dim, keepdims=keepdims)
    x = x + nobj.log().view([-1] + [1, ] * (len(x.shape) - 1))
    return x


def masked_var(x, nobj, dim=None, keepdims=False):
    var = (masked_mean((x - masked_mean(x, nobj, dim, keepdims=True)) ** 2, nobj, dim, keepdims))
    return var


def masked_sum(x, N, dim=None, keepdims=False):
    if type(dim) != int:
        N = N ** (len(dim))
    return x.sum(dim=dim, keepdims=keepdims) / N


# --- Equivariant operations ---
# Notation: eops_r_to_s maps rank-r tensors to rank-s tensors equivariantly under Sₙ.
#
# In HERON:
#   rank-1 tensor = per-asset feature vector, shape [B, N, C]
#   rank-2 tensor = pairwise asset feature matrix, shape [B, N, N, C]
#   rank-0 tensor = portfolio-level scalar, shape [B, C]
#
# The 15-basis Eq₂→₂ is the main workhorse: it propagates pairwise correlation/interaction
# information through the network while respecting the "assets are exchangeable" symmetry.

def eops_1_to_1(inputs, normalize=False):
    """Per-asset features → per-asset features (2 basis elements)."""
    inputs = inputs.permute(0, 2, 1)
    dim = inputs.shape[-1]
    sums = inputs.sum(dim=-1, keepdim=True) / dim
    op1 = inputs
    op2 = sums.expand(-1, -1, dim)
    return torch.stack([op1, op2], dim=2)


def eops_1_to_2(inputs, nobj=None, nobj_avg=100, aggregation='mean', weight=None):
    """
    Per-asset features → pairwise features (5 basis elements).

    In HERON this promotes individual asset signals (momentum, volatility, etc.)
    into the pairwise space so they can interact with correlation-based features.
    nobj_avg: typical universe size (default 100 for equity universes).
    """
    inputs = inputs.permute(0, 2, 1)
    B, C, N = inputs.shape

    if aggregation == 'mean':
        aggregation_fn = masked_mean
    elif aggregation == 'max':
        aggregation_fn = masked_amax
    elif aggregation == 'min':
        aggregation_fn = masked_amin
    elif aggregation == 'var':
        aggregation_fn = masked_var
    elif aggregation == 'sum':
        aggregation_fn = masked_sum
        nobj = nobj_avg

    if weight is not None:
        sum_all = aggregation_fn(inputs * weight.unsqueeze(1), nobj, dim=2, keepdims=True)
    else:
        sum_all = aggregation_fn(inputs, nobj, dim=2, keepdims=True)

    op1 = torch.diag_embed(inputs)
    op2 = inputs.unsqueeze(2).expand(-1, -1, N, -1)
    op3 = inputs.unsqueeze(3).expand(-1, -1, -1, N)
    op4 = torch.diag_embed(sum_all.expand(-1, -1, N))
    op5 = sum_all.unsqueeze(3).expand(-1, -1, N, N)
    return torch.stack([op1, op2, op3, op4, op5], dim=2)


def eops_2_to_0(inputs, nobj=None, nobj_avg=100, aggregation='mean', weight=None):
    """
    Pairwise features → portfolio scalar (2 basis elements: total sum, diagonal sum).

    In HERON's Eq₂→₀ output head, this aggregates the N×N pairwise representation
    down to portfolio-level invariants used for regime classification or portfolio-level
    predictions. The two aggregators correspond to total market interaction (sum_all)
    and self-interaction terms (sum_diag, analogous to per-asset variances).
    """
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    if aggregation == 'mean':
        aggregation_fn = masked_mean
    elif aggregation == 'max':
        aggregation_fn = masked_amax
    elif aggregation == 'min':
        aggregation_fn = masked_amin
    elif aggregation == 'var':
        aggregation_fn = masked_var
    elif aggregation == 'sum':
        aggregation_fn = masked_sum
        nobj = nobj_avg

    if weight is not None:
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2)
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2, 3))
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2)
        sum_all = aggregation_fn(inputs, nobj, dim=(2, 3))

    ops = [sum_all, sum_diag_part]
    return torch.stack(ops, dim=2)


def eops_2_to_1(inputs, nobj=None, nobj_avg=100, aggregation='mean', weight=None):
    """
    Pairwise features → per-asset features (5 basis elements).

    In HERON's Eq₂→₁ output head, this maps the N×N pairwise representation to
    a per-asset score vector. The per-asset MLP then converts this to a predicted
    return or alpha score for each asset. Permuting assets permutes the scores
    identically — the model never implicitly learns from asset ordering.
    """
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    if aggregation == 'mean':
        aggregation_fn = masked_mean
    elif aggregation == 'max':
        aggregation_fn = masked_amax
    elif aggregation == 'min':
        aggregation_fn = masked_amin
    elif aggregation == 'var':
        aggregation_fn = masked_var
    elif aggregation == 'sum':
        aggregation_fn = masked_sum
        nobj = nobj_avg

    if weight is not None:
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs * weight_rows, nobj, dim=3)
        sum_cols = aggregation_fn(inputs * weight_cols, nobj, dim=2)
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2, 3))
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs, nobj, dim=3)
        sum_cols = aggregation_fn(inputs, nobj, dim=2)
        sum_all = aggregation_fn(inputs, nobj, dim=(2, 3))

    op1 = diag_part
    op2 = sum_rows
    op3 = sum_cols
    op4 = sum_diag_part.expand(-1, -1, N)
    op5 = sum_all.unsqueeze(2).expand(-1, -1, N)
    ops = [op1, op2, op3, op4, op5]
    return torch.stack(ops, dim=2)


def eops_2_to_2(inputs, nobj=None, nobj_avg=100, aggregation='mean', weight=None,
                skip_order_zero=False, folklore=False):
    """
    Pairwise features → pairwise features (15 basis elements).

    This is the main message-passing operation in HERON. It takes the N×N matrix of
    pairwise asset interactions and produces a new N×N matrix, mixing information
    between asset pairs while preserving permutation equivariance.

    The 15 basis elements (Pan & Kondor, 2022) include:
      - 5 "skip" ops: identity, transpose, diagonal embed, and row/col broadcast of diagonal
      - 10 "mixing" ops: row/col sums broadcast as rows, cols, diagonals, and global sum

    HERON uses the full 15-element basis, since no symmetry constraint beyond
    permutation applies to this input space — giving the model substantial
    expressive power for its parameter budget.

    The learnable (N/N̄)^α scaling exponent is critical for finance: it lets the model
    trained on 300-stock universes deploy cleanly on 500-stock universes without retraining.
    """
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    if aggregation == 'mean':
        aggregation_fn = masked_mean
    elif aggregation == 'max':
        aggregation_fn = masked_amax
    elif aggregation == 'min':
        aggregation_fn = masked_amin
    elif aggregation == 'var':
        aggregation_fn = masked_var
    elif aggregation == 'sum':
        aggregation_fn = masked_sum
        nobj = nobj_avg

    if weight is not None:
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs * weight_rows, nobj, dim=3)
        sum_cols = aggregation_fn(inputs * weight_cols, nobj, dim=2)
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2, 3))
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs, nobj, dim=3)
        sum_cols = aggregation_fn(inputs, nobj, dim=2)
        sum_all = aggregation_fn(inputs, nobj, dim=(2, 3))

    ops = [None] * (17 if folklore else 16)

    if not skip_order_zero:
        ops[1] = inputs                                                         # identity
        ops[2] = torch.transpose(inputs, 2, 3)                                  # transpose
        ops[3] = torch.diag_embed(diag_part)                                    # zero off-diagonal
        ops[4] = diag_part.unsqueeze(2).expand(-1, -1, N, -1)                   # broadcast diag as rows
        ops[5] = diag_part.unsqueeze(3).expand(-1, -1, -1, N)                   # broadcast diag as cols

    ops[6] = torch.diag_embed(sum_cols)
    ops[7] = sum_cols.unsqueeze(2).expand(-1, -1, N, -1)
    ops[8] = sum_cols.unsqueeze(3).expand(-1, -1, -1, N)
    ops[9] = torch.diag_embed(sum_rows)
    ops[10] = sum_rows.unsqueeze(2).expand(-1, -1, N, -1)
    ops[11] = sum_rows.unsqueeze(3).expand(-1, -1, -1, N)
    ops[12] = sum_diag_part.unsqueeze(3).expand(-1, -1, N, N)
    ops[13] = torch.diag_embed(sum_diag_part.expand(-1, -1, N))
    ops[14] = sum_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, N)
    ops[15] = torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, N))

    if folklore:
        ops[16] = aggregation_fn(
            torch.nn.LeakyReLU()(inputs.unsqueeze(-2) + inputs.unsqueeze(-3).permute(0, 1, 2, 4, 3)),
            nobj, dim=-1)

    if skip_order_zero:
        ops = torch.stack(ops[6:], dim=2)
    else:
        ops = torch.stack(ops[1:], dim=2)

    return ops

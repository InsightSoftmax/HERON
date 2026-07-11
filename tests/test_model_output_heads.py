"""Forward-shape and symmetry checks for both HERON output heads.

Also documents a known gap (see test_portfolio_head_has_no_working_loss):
the 'portfolio' head is wired up in the model but has no compatible loss
function in the trainer. If that test starts failing, the loss has been
fixed and this test (and the gap noted in project memory) should be updated.
"""
import pytest
import torch

from src.models.heron import HERON
from src.trainer.trainer import sharpe_loss

from conftest import TINY_MODEL_KWARGS

B, N = 2, 20


def make_batch(seed=0):
    torch.manual_seed(seed)
    feats = torch.randn(B, N, 7)
    corr = torch.randn(B, N, N)
    corr = (corr + corr.transpose(1, 2)) / 2
    mask = torch.ones(B, N, dtype=torch.bool)
    edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    nobj = torch.full((B, 1), N, dtype=torch.float32)
    return {
        'features': feats,
        'correlations': corr,
        'particle_mask': mask,
        'edge_mask': edge_mask,
        'nobj': nobj,
    }


def permute_batch(data, perm):
    return {
        'features': data['features'][:, perm, :],
        'correlations': data['correlations'][:, perm, :][:, :, perm],
        'particle_mask': data['particle_mask'][:, perm],
        'edge_mask': data['edge_mask'][:, perm, :][:, :, perm],
        'nobj': data['nobj'],
    }


def test_crosssectional_head_output_shape():
    model = HERON(output_head='crosssectional', **TINY_MODEL_KWARGS)
    model.eval()
    with torch.no_grad():
        out = model(make_batch())['predict']
    assert out.shape == (B, N)


def test_portfolio_head_output_shape():
    model = HERON(output_head='portfolio', **TINY_MODEL_KWARGS)
    model.eval()
    with torch.no_grad():
        out = model(make_batch())['predict']
    assert out.shape == (B, 2)


def test_crosssectional_head_is_permutation_equivariant():
    torch.manual_seed(1)
    model = HERON(output_head='crosssectional', **TINY_MODEL_KWARGS)
    model.eval()

    data = make_batch()
    perm = torch.randperm(N)
    data_perm = permute_batch(data, perm)

    with torch.no_grad():
        out = model(data)['predict']
        out_perm = model(data_perm)['predict']

    torch.testing.assert_close(out[:, perm], out_perm, atol=1e-5, rtol=1e-5)


def test_portfolio_head_is_permutation_invariant():
    torch.manual_seed(1)
    model = HERON(output_head='portfolio', **TINY_MODEL_KWARGS)
    model.eval()

    data = make_batch()
    perm = torch.randperm(N)
    data_perm = permute_batch(data, perm)

    with torch.no_grad():
        out = model(data)['predict']
        out_perm = model(data_perm)['predict']

    torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=1e-5)


def test_portfolio_head_has_no_working_loss():
    """Known gap: sharpe_loss hard-assumes [B, N] per-asset predictions.

    The portfolio head returns [B, 2], so training with
    --output-head=portfolio currently crashes. This test pins that
    behaviour down; if someone adds a portfolio-compatible loss, this
    test (and the note in project memory) needs to be updated together
    with the trainer wiring.
    """
    model = HERON(output_head='portfolio', **TINY_MODEL_KWARGS)
    model.eval()

    data = make_batch()
    with torch.no_grad():
        predictions = model(data)['predict']  # [B, 2]

    targets = torch.randn(B, N)  # per-asset targets, mismatched with [B, 2]
    with pytest.raises(RuntimeError):
        sharpe_loss(predictions, targets, mask=data['particle_mask'])

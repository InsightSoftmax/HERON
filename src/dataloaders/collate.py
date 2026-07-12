"""
collate.py — HERON batch collation

Pads variable-size cross-sections to a common universe size within a batch,
and builds the masks that tell the equivariant layers which asset slots are
real vs. padding. Masking is based on nobj or a dedicated mask field —
zero-padded asset slots are excluded from all aggregation operations.

The mask mechanism is essential for variable universe sizes: if a training batch
contains cross-sections of sizes [300, 287, 315], all are padded to 315, and the
mask tells the equivariant layers to ignore the padded slots.
"""

import torch
import numpy as np


def batch_stack(props, edge_mat=False, nobj=None):
    """
    Stack a list of tensors, padding to the largest size in the batch.

    Parameters
    ----------
    props : list of Tensor
    edge_mat : bool
        If True, pads along both axes (for N×N correlation/pairwise matrices).
    nobj : int or None
        If set, truncate to at most nobj assets per cross-section.
    """
    if nobj is not None and nobj < 0:
        nobj = None
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        props = [p[:nobj, ...] for p in props]
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_assets = max([len(p) for p in props])
        if nobj is not None:
            max_assets = min(max_assets, nobj)
        max_shape = (len(props), max_assets, max_assets) + props[0].shape[2:]
        padded = torch.zeros(max_shape, dtype=props[0].dtype)
        for idx, prop in enumerate(props):
            n = min(len(prop), max_assets)
            padded[idx, :n, :n] = prop[:n, :n]
        return padded


def batch_stack_general(props):
    """Stack tensors of potentially varying shapes, auto-detecting scalars vs vectors vs matrices."""
    if type(props[0]) in [int, float]:
        return torch.tensor(props)
    if type(props[0]) is np.ndarray:
        props = [torch.from_numpy(p) for p in props]
    shapes = [p.shape for p in props]
    if all(shapes[0] == s for s in shapes):
        return torch.stack(props)
    elif all(shapes[0][1:] == s[1:] for s in shapes):
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    elif all(shapes[0][2:] == s[2:] for s in shapes):
        assert all(s[0] == s[1] for s in shapes), 'Pairwise matrix must be square'
        max_n = max(len(p) for p in props)
        padded = torch.zeros((len(props), max_n, max_n) + props[0].shape[2:], dtype=props[0].dtype)
        for idx, prop in enumerate(props):
            n = len(prop)
            padded[idx, :n, :n] = prop
        return padded
    else:
        raise ValueError('Cannot batch tensors with incompatible shapes')


def collate_fn(data, nobj=None):
    """
    Collate a list of EquityDataset items into a batch.

    Produces:
      features        : [B, N_max, d_f]   per-asset feature matrix
      correlations    : [B, N_max, N_max]  pairwise correlation matrix
      targets         : [B, N_max]         per-asset forward return targets
      particle_mask   : [B, N_max]         True for real assets, False for padding
      edge_mask       : [B, N_max, N_max]  True where both assets are real
      nobj            : [B, 1]             actual number of assets per cross-section
    """
    if data[0] is None:
        return None

    common_keys = data[0].keys()
    batch = {key: batch_stack(
        [event[key] for event in data],
        edge_mat=(key == 'correlations'),
        nobj=nobj
    ) for key in common_keys}

    # Build masks from nobj field or from feature zero-rows
    if 'nobj' in batch:
        n_assets = batch['nobj'].long()  # [B]
        B, N_max = batch['features'].shape[:2]
        idx = torch.arange(N_max).unsqueeze(0).expand(B, -1)  # [B, N_max]
        particle_mask = idx < n_assets.unsqueeze(1)            # [B, N_max]
    else:
        # Fall back: treat rows with all-zero features as padding
        particle_mask = (batch['features'].abs().sum(-1) != 0)

    edge_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)

    batch['particle_mask'] = particle_mask.bool()
    batch['edge_mask'] = edge_mask.bool()

    # Ensure nobj is stored as [B, 1] for use in equivariant scaling
    if 'nobj' not in batch:
        batch['nobj'] = particle_mask.sum(-1, keepdim=True).float()
    else:
        batch['nobj'] = batch['nobj'].unsqueeze(-1).float()

    return batch

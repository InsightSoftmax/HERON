"""
financial_features.py — HERON input feature computation

The fundamental pairwise inputs to HERON are financial pairwise signals:
    A_ij = f(feature_i, feature_j)

where A_ij can be:
  - Rolling Pearson correlation of returns: ρ_ij
  - Learned bilinear interactions: f_i^T W^k f_j
  - Co-skewness estimates
  - Factor cross-exposures: β_i^T Σ_F β_j

The only symmetry constraint on this input space is permutation invariance
(the symmetric group Sₙ), so all 15 Eq₂→₂ basis elements are admissible,
giving the model substantial expressive power.

A learnable power-law encoding f_β(x) = ((1+x)^β² − 1)/β² is applied to these
pairwise signals: financial correlations have a "peaked near zero, fat-tailed"
distribution, and this encoding handles that shape effectively.
"""

import torch
import torch.nn as nn
import numpy as np


class FinancialPairwiseFeatures(nn.Module):
    """
    Computes the N×N pairwise feature tensor from per-asset feature vectors.

    HERON computes:

      1. Pre-computed correlations loaded directly from data (ρ_ij)
      2. Learned bilinear interactions f_i^T W^k f_j (num_bilinear of these)

    The output is a rank-2 tensor A of shape [B, N, N, rank2_dim] that feeds
    into the InputEncoder and then the equivariant Net2to2 blocks.

    Parameters
    ----------
    feature_dim : int
        Dimension of per-asset feature vectors (d_f).
    num_bilinear : int
        Number of learned bilinear interaction channels (W^k matrices).
        Set to 0 to use only pre-computed correlations.
    use_precomputed_corr : bool
        Whether to expect a pre-computed correlation matrix in the input data.
    """

    def __init__(self, feature_dim, num_bilinear=4, use_precomputed_corr=True,
                 device=torch.device('cpu'), dtype=torch.float):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bilinear = num_bilinear
        self.use_precomputed_corr = use_precomputed_corr

        # rank2_dim: number of channels in the pairwise feature tensor
        self.rank2_dim = (1 if use_precomputed_corr else 0) + num_bilinear
        # rank1_dim: per-asset features are handled separately via Eq1to2
        self.rank1_dim = 0

        if num_bilinear > 0:
            # Learned projection matrices W^k: f_i^T W^k f_j
            # Each W^k is a feature_dim × feature_dim matrix
            self.bilinear_weights = nn.ParameterList([
                nn.Parameter(torch.randn(feature_dim, feature_dim, device=device, dtype=dtype)
                             * np.sqrt(1.0 / feature_dim))
                for _ in range(num_bilinear)
            ])

        self.to(device=device, dtype=dtype)

    def forward(self, asset_features, correlations=None):
        """
        Parameters
        ----------
        asset_features : Tensor [B, N, d_f]
            Per-asset feature vectors (returns, volatility, momentum, etc.).
        correlations : Tensor [B, N, N] or None
            Pre-computed rolling return correlations. Required if use_precomputed_corr=True.

        Returns
        -------
        rank2_inputs : Tensor [B, N, N, rank2_dim]
            Pairwise feature tensor.
        """
        channels = []

        if self.use_precomputed_corr:
            assert correlations is not None, "correlations must be provided when use_precomputed_corr=True"
            # Shape: [B, N, N, 1]
            channels.append(correlations.unsqueeze(-1))

        for W in self.bilinear_weights:
            # f_i^T W f_j: project features then take outer product
            # projected: [B, N, d_f]
            projected = asset_features @ W
            # bilinear[b, i, j] = projected[b, i, :] · asset_features[b, j, :]
            bilinear = torch.bmm(projected, asset_features.transpose(1, 2))  # [B, N, N]
            channels.append(bilinear.unsqueeze(-1))

        rank2_inputs = torch.cat(channels, dim=-1)  # [B, N, N, rank2_dim]
        return rank2_inputs


class FinancialInputEncoder(nn.Module):
    """
    Learnable encoding of raw financial pairwise signals:
        f_β(x) = ((1 + x)^β² − 1) / β²

    Financial correlations and bilinear signals have a heavy-tailed,
    non-Gaussian distribution, and this power-law encoding handles it
    effectively.

    For correlations ρ ∈ [−1, 1], we use the signed variant:
        g_β(ρ) = ((1 + |ρ|)^β² − 1) / β² × sign(ρ)

    Multiple trainable β parameters span a range of sensitivity scales,
    allowing the network to use both fine and coarse correlation signals.

    Parameters
    ----------
    out_dim : int
        Number of output channels per input pairwise signal.
    rank2_in_dim : int
        Number of input pairwise channels (rank2_dim from FinancialPairwiseFeatures).
    """

    def __init__(self, out_dim, rank2_in_dim=1, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()
        self.rank2_in_dim = rank2_in_dim
        self.out_dim = out_dim

        # Multiple β values, initialised to span [0.1, 0.5]
        self.rank2_alphas = nn.Parameter(
            torch.linspace(0.05, 0.5, out_dim, device=device, dtype=dtype)
        )
        self.zero = torch.tensor(0, device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)

    def forward(self, rank2_inputs, rank2_mask=None):
        """
        Parameters
        ----------
        rank2_inputs : Tensor [B, N, N, rank2_in_dim]
        rank2_mask : BoolTensor [B, N, N, 1] or None

        Returns
        -------
        encoded : Tensor [B, N, N, out_dim]
        """
        # Signed power-law encoding: handles negative correlations correctly
        alphas = self.rank2_alphas.view([1, 1, 1, self.out_dim])
        x = rank2_inputs[..., :1]  # Use first channel (correlation) for encoding
        encoded = ((1 + x.abs()).pow(1e-6 + alphas ** 2) - 1) / (1e-6 + alphas ** 2) * x.sign()

        if rank2_mask is not None:
            encoded = torch.where(rank2_mask, encoded, self.zero)

        return encoded


def compute_rolling_correlation(returns, window=63, min_periods=20):
    """
    Compute rolling Pearson correlation matrix from a returns time series.

    Utility function for data preprocessing — not used inside the model itself.

    Parameters
    ----------
    returns : np.ndarray [T, N]
        Daily returns for N assets over T periods.
    window : int
        Rolling window in trading days (default: 63 ≈ 3 months).
    min_periods : int
        Minimum number of valid observations required.

    Returns
    -------
    corr_matrices : np.ndarray [T, N, N]
        Rolling correlation matrix at each time step.
    """
    import numpy as np
    import pandas as pd
    T, N = returns.shape
    df = pd.DataFrame(returns)
    corr_matrices = np.zeros((T, N, N))
    for t in range(T):
        start = max(0, t - window + 1)
        window_data = df.iloc[start:t + 1]
        if len(window_data) >= min_periods:
            corr = window_data.corr().values
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            corr = np.zeros((N, N))
        corr_matrices[t] = corr
    return corr_matrices

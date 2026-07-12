"""
heron.py — HERON model

Hierarchical Equivariant Return Optimization Network

Architecture overview
---------------------
Input pipeline:
    asset features + correlations → FinancialPairwiseFeatures → FinancialInputEncoder
                                  → Net2to2 → Eq2to1/Eq2to0 → per-asset MLP / portfolio head

Two output heads:
  - Eq2to1 head: per-asset alpha scores → cross-sectional return prediction
  - Eq2to0 head: portfolio-level prediction → regime classification or portfolio weights

Market context:
  HERON can prepend a small number of "market assets" to the asset list. These encode
  index-level signals (market return, VIX, yield curve slope) and interact with all
  real assets through the equivariant layers without contaminating the permutation
  equivariance of the real asset outputs.
"""

import torch
import torch.nn as nn
import logging

from ..layers import BasicMLP, Net2to2, Eq1to2, Eq2to0, Eq2to1, MessageNet, MyLinear
from ..layers import InputEncoder
from .financial_features import FinancialPairwiseFeatures, FinancialInputEncoder
from ..trainer.utils import init_weights

logger = logging.getLogger(__name__)


class HERON(nn.Module):
    """
    Hierarchical Equivariant Return Optimization Network.

    Parameters
    ----------
    feature_dim : int
        Dimension of per-asset input features d_f.
        E.g. 7 for (ret_5d, ret_21d, ret_63d, ret_252d, vol_21d, vol_63d, turnover).
    num_bilinear : int
        Number of learned bilinear pairwise channels (W^k matrices).
    num_channels_scalar : int
        Output channels from the Eq1to2 promotion of per-asset features.
    num_channels_m : list of list of int
        Channel dims for MessageNet sub-blocks in Net2to2.
        E.g. [[35], [35], [35], [35], [35]]
    num_channels_2to2 : list of int
        Channel dims for Eq2to2 layers.
        E.g. [60, 60, 60, 60, 60]
    num_channels_out : list of int
        Channel dims for the output MLP.
    num_channels_m_out : list of int
        Channel dims for the final MessageNet before aggregation.
    output_head : str
        'crosssectional' (Eq2to1 → per-asset scores) or
        'portfolio'      (Eq2to0 → portfolio weights).
    num_market_features : int
        Number of market context slots prepended to the asset list (0 to disable).
    average_nobj : int
        Typical universe size N̄. Used in the (N/N̄)^α learnable scaling.
        Setting this correctly is important for generalisation across universe sizes.
    activation : str
        Activation function ('leakyrelu', 'gelu', etc.).
    config : str
        Aggregation config string for Net2to2 ('s'=sum, 'm'=mean, 'S'=scaled sum, etc.).
    dropout, drop_rate : bool, float
        Dropout settings.
    batchnorm : str or None
        Batch normalisation type ('b'=BatchNorm, 'i'=InstanceNorm, None=off).
    """

    def __init__(
            self,
            feature_dim,
            num_bilinear=4,
            num_channels_scalar=16,
            num_channels_m=None,
            num_channels_2to2=None,
            num_channels_out=None,
            num_channels_m_out=None,
            output_head='crosssectional',
            num_market_features=0,
            average_nobj=100,
            activate_agg=False,
            activate_lin=True,
            activate_agg_out=True,
            activate_lin_out=False,
            activation='leakyrelu',
            config='S',
            config_out='S',
            factorize=False,
            masked=True,
            mlp_out=True,
            dropout=False,
            drop_rate=0.1,
            drop_rate_out=0.1,
            batchnorm=None,
            device=torch.device('cpu'),
            dtype=torch.float,
    ):
        super().__init__()

        # Defaults matching HERON's best-performing configuration
        if num_channels_m is None:
            num_channels_m = [[35]] * 5
        if num_channels_2to2 is None:
            num_channels_2to2 = [60, 60, 60, 60, 60]
        if num_channels_out is None:
            num_channels_out = [64, 32]
        if num_channels_m_out is None:
            num_channels_m_out = [60]

        self.device = device
        self.dtype = dtype
        self.feature_dim = feature_dim
        self.output_head = output_head
        self.num_market_features = num_market_features
        self.average_nobj = average_nobj
        self.mlp_out = mlp_out
        self.dropout = dropout
        self.config = config
        self.config_out = config_out

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        # --- Input stage ---
        # Step 1: Compute pairwise features (correlations + bilinear interactions)
        self.pairwise_features = FinancialPairwiseFeatures(
            feature_dim=feature_dim,
            num_bilinear=num_bilinear,
            use_precomputed_corr=True,
            device=device, dtype=dtype,
        )
        rank2_dim = self.pairwise_features.rank2_dim

        # Determine embedding dimension (first channel of Net2to2 input)
        if len(num_channels_m) > 0 and len(num_channels_m[0]) > 0:
            embedding_dim = num_channels_m[0][0]
        else:
            embedding_dim = num_channels_2to2[0]

        # Reserve channels for Eq1to2 output if we have per-asset features
        pairwise_embedding_dim = embedding_dim - num_channels_scalar

        # Step 2: Encode pairwise features with learnable power-law encoding
        self.input_encoder = FinancialInputEncoder(
            out_dim=pairwise_embedding_dim,
            rank2_in_dim=rank2_dim,
            device=device, dtype=dtype,
        )

        # Step 3: Promote per-asset features to pairwise via Eq1to2
        total_per_asset_dim = feature_dim + num_market_features
        self.eq1to2 = Eq1to2(
            total_per_asset_dim, num_channels_scalar,
            activate_agg=activate_agg, activate_lin=activate_lin,
            activation=activation, average_nobj=average_nobj,
            config=config_out, factorize=False,
            device=device, dtype=dtype,
        )

        # --- Equivariant core ---
        # Net2to2: stack of 5 Eq2to2 blocks (message + aggregation)
        self.net2to2 = Net2to2(
            num_channels_2to2 + [num_channels_m_out[0]],
            num_channels_m,
            activate_agg=activate_agg, activate_lin=activate_lin,
            activation=activation,
            dropout=dropout, drop_rate=drop_rate,
            batchnorm=batchnorm, config=config,
            average_nobj=average_nobj, factorize=factorize,
            masked=masked, device=device, dtype=dtype,
        )

        # --- Output heads ---
        self.msg_out = MessageNet(
            num_channels_m_out, activation=activation,
            batchnorm=batchnorm, device=device, dtype=dtype,
        )

        if output_head == 'crosssectional':
            # Eq2to1: N×N tensor → per-asset vector → per-asset alpha score
            # Permuting assets permutes scores identically (equivariant output)
            self.agg_out = Eq2to1(
                num_channels_m_out[-1],
                num_channels_out[0] if mlp_out else 1,
                activate_agg=activate_agg_out, activate_lin=activate_lin_out,
                activation=activation, config=config_out, factorize=False,
                average_nobj=average_nobj, device=device, dtype=dtype,
            )
            if mlp_out:
                self.mlp_out_layer = BasicMLP(
                    num_channels_out + [1],
                    activation=activation, dropout=False,
                    batchnorm=False, device=device, dtype=dtype,
                )

        elif output_head == 'portfolio':
            # Eq2to0: N×N tensor → invariant scalars → softmax portfolio weights
            self.agg_out = Eq2to0(
                num_channels_m_out[-1],
                num_channels_out[0] if mlp_out else 2,
                activate_agg=activate_agg_out, activate_lin=activate_lin_out,
                activation=activation, config=config_out, factorize=False,
                average_nobj=average_nobj, device=device, dtype=dtype,
            )
            if mlp_out:
                self.mlp_out_layer = BasicMLP(
                    num_channels_out + [2],
                    activation=activation, dropout=False,
                    batchnorm=False, device=device, dtype=dtype,
                )

        self.apply(init_weights)

        logger.info('HERON initialised.')
        logger.info('_________________________')
        for n, p in self.named_parameters():
            logger.info(f'{"Parameter: " + n:<80} {p.shape}')
        logger.info(f'Total parameters: {sum(p.nelement() for p in self.parameters()):,}')
        logger.info('_________________________')

    def forward(self, data, return_intermediates=False):
        """
        Forward pass.

        Parameters
        ----------
        data : dict with keys:
            'features'      : [B, N, d_f]   per-asset features
            'correlations'  : [B, N, N]      pairwise correlation matrix
            'particle_mask' : [B, N]         bool mask
            'edge_mask'     : [B, N, N]      bool mask
            'nobj'          : [B, 1]         number of real assets
            'market_features' (optional)     [B, d_m] market context

        Returns
        -------
        dict with 'predict':
            cross-sectional head: [B, N] per-asset alpha scores
            portfolio head: [B, 2] long/short weights logits
        """
        asset_features, correlations, particle_mask, edge_mask, nobj = self.prepare_input(data)

        # Add market context features if provided
        if self.num_market_features > 0 and 'market_features' in data:
            asset_features, particle_mask, edge_mask, nobj = self._prepend_market_particles(
                asset_features, data['market_features'], particle_mask, edge_mask, nobj
            )

        # --- Input encoding ---
        # Compute pairwise financial features (correlations + bilinear)
        rank2_raw = self.pairwise_features(asset_features, correlations)  # [B, N, N, rank2_dim]

        # Encode with learnable power-law (handles heavy-tailed correlation distribution)
        rank2_encoded = self.input_encoder(
            rank2_raw, rank2_mask=edge_mask.unsqueeze(-1)
        )  # [B, N, N, pairwise_embedding_dim]

        # Promote per-asset features to pairwise via Eq1to2
        rank2_from_rank1 = self.eq1to2(
            asset_features,
            mask=edge_mask.unsqueeze(-1),
            nobj=nobj,
        )  # [B, N, N, num_channels_scalar]

        # Concatenate: pairwise-encoded + promoted per-asset features
        inputs = torch.cat([rank2_encoded, rank2_from_rank1], dim=-1)  # [B, N, N, embedding_dim]

        # --- Equivariant core ---
        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj)

        # --- Output head ---
        act2 = self.msg_out(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)

        act3 = self.agg_out(act2, nobj=nobj)

        if self.dropout:
            act3 = self.dropout_layer_out(act3)

        if self.mlp_out:
            prediction = self.mlp_out_layer(act3)
        else:
            prediction = act3

        # For cross-sectional head: squeeze last dim to get [B, N] scores
        if self.output_head == 'crosssectional':
            prediction = prediction.squeeze(-1)  # [B, N]
            # Mask out padded asset slots
            prediction = prediction * particle_mask.float()

        assert not torch.isnan(prediction).any(), "NaN values in HERON output."

        if return_intermediates:
            return {'predict': prediction, 'inputs': inputs, 'act1': act1, 'act2': act2, 'act3': act3}
        return {'predict': prediction}

    def prepare_input(self, data):
        device, dtype = self.device, self.dtype
        asset_features = data['features'].to(device, dtype)
        correlations = data['correlations'].to(device, dtype)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)
        nobj = data['nobj'].to(device, dtype)
        return asset_features, correlations, particle_mask, edge_mask, nobj

    def _prepend_market_particles(self, asset_features, market_features, particle_mask, edge_mask, nobj):
        """
        Prepend market context slots to the asset list.

        These provide information about the ambient market environment that
        is invisible to a purely cross-sectional model, e.g. the market
        return, VIX level, or yield curve slope.

        The market context slots are flagged so the Eq2to1 output head can
        exclude them from the per-asset prediction vector.
        """
        B, N, d_f = asset_features.shape
        device, dtype = self.device, self.dtype

        # Encode market features as per-asset rows prepended to asset list
        d_m = market_features.shape[-1]
        # Pad or project market features to match asset feature dimension
        if d_m < d_f:
            pad = torch.zeros(B, 1, d_f - d_m, device=device, dtype=dtype)
            mkt_row = torch.cat([market_features.unsqueeze(1), pad], dim=-1)
        else:
            mkt_row = market_features[:, :d_f].unsqueeze(1)

        # Prepend (num_market_features=1 for simplicity here)
        asset_features = torch.cat([mkt_row, asset_features], dim=1)

        # Extend masks
        mkt_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
        particle_mask = torch.cat([mkt_mask, particle_mask], dim=1)
        edge_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)
        nobj = nobj + 1

        return asset_features, particle_mask, edge_mask, nobj


def expand_var_list(var):
    if type(var) is list:
        return var
    raise ValueError(f'Expected list, got {type(var)}')

#!/usr/bin/env python3
"""
generate_synthetic.py — Synthetic equity data generator for HERON

Generates a realistic synthetic cross-sectional equity dataset using a
factor model with:
  - K latent risk factors with time-varying returns
  - N assets with time-varying factor loadings (AR(1) processes)
  - Regime-switching correlations (bull/bear/crisis regimes)
  - Heavy-tailed idiosyncratic returns (Student-t noise)

This mirrors the "factor-model synthetic dataset" described in the HERON paper
(Section 5.1) and is suitable for validating the architecture before applying
it to proprietary real data.

Output: HDF5 files (train.h5, valid.h5, test.h5) in a format directly
        consumable by EquityDataset.

Usage:
    python data/generate_synthetic.py --n-assets 200 --n-days 2000 --outdir data/sample_data
    python data/generate_synthetic.py --n-assets 500 --n-days 3000 --outdir data/large
"""

import numpy as np
import h5py
import argparse
import os
from datetime import datetime, timedelta


def generate_factor_model_data(
        n_assets=200,
        n_days=2000,
        n_factors=5,
        seed=42,
        feature_window_days=(5, 21, 63, 252),
        vol_window_days=(21, 63),
        corr_window=63,
        fwd_return_horizon=21,
):
    """
    Generate synthetic cross-sectional equity data.

    Returns a dict of arrays indexed by cross-section date (observation t):
      features     : [T, N, d_f]   per-asset feature vectors
      correlations : [T, N, N]     rolling return correlation matrices
      targets      : [T, N]        forward rank-normalised returns
      nobj         : [T]           number of real assets (constant here)
      dates        : [T]           date identifiers (YYYYMMDD int)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Factor returns — regime-switching                                 #
    # ------------------------------------------------------------------ #
    # Regime: 0=bull (low vol), 1=bear (high vol), 2=crisis (very high vol)
    regime_probs = np.array([0.60, 0.30, 0.10])
    regime_vols = np.array([0.008, 0.015, 0.030])  # daily factor vol by regime

    regimes = rng.choice(3, size=n_days, p=regime_probs)
    # Smooth regime transitions
    for t in range(1, n_days):
        if rng.random() > 0.03:  # stay in regime with 97% probability
            regimes[t] = regimes[t - 1]

    factor_returns = np.zeros((n_days, n_factors))
    for t in range(n_days):
        vol = regime_vols[regimes[t]]
        factor_returns[t] = rng.normal(0, vol, n_factors)
    # Factor means: market ~+5% annualised, others ~0
    factor_returns[:, 0] += 0.05 / 252

    # ------------------------------------------------------------------ #
    # 2. Asset factor loadings — AR(1) time-varying betas                 #
    # ------------------------------------------------------------------ #
    # Each asset has K factor loadings that evolve slowly over time
    beta_mean = rng.normal(0, 0.5, (n_assets, n_factors))
    beta_mean[:, 0] = np.abs(beta_mean[:, 0])  # market beta > 0
    betas = np.zeros((n_days, n_assets, n_factors))
    betas[0] = beta_mean
    for t in range(1, n_days):
        noise = rng.normal(0, 0.005, (n_assets, n_factors))
        betas[t] = 0.99 * betas[t - 1] + 0.01 * beta_mean + noise

    # ------------------------------------------------------------------ #
    # 3. Asset returns                                                     #
    # ------------------------------------------------------------------ #
    # Systematic return + idiosyncratic (Student-t for fat tails)
    idio_vol = 0.015  # ~24% annualised idiosyncratic vol
    idio_df = 5       # degrees of freedom for fat tails
    systematic = np.einsum('ti,ki->tk', factor_returns, betas.mean(0))  # simplified
    # Use actual time-varying betas
    systematic = np.array([
        factor_returns[t] @ betas[t].T for t in range(n_days)
    ])  # [T, N]
    idio = rng.standard_t(idio_df, size=(n_days, n_assets)) * idio_vol / np.sqrt(idio_df / (idio_df - 2))
    asset_returns = systematic + idio  # [T, N]

    # ------------------------------------------------------------------ #
    # 4. Build features and targets for each cross-section                #
    # ------------------------------------------------------------------ #
    warmup = max(feature_window_days) + corr_window + fwd_return_horizon
    T = n_days - warmup - fwd_return_horizon
    if T <= 0:
        raise ValueError(f'Not enough days ({n_days}) for warmup ({warmup}) + T + horizon ({fwd_return_horizon})')

    d_f = len(feature_window_days) + len(vol_window_days) + 1  # returns + vols + turnover proxy

    features_out = np.zeros((T, n_assets, d_f), dtype=np.float32)
    correlations_out = np.zeros((T, n_assets, n_assets), dtype=np.float32)
    targets_out = np.zeros((T, n_assets), dtype=np.float32)
    dates_out = np.zeros(T, dtype=np.int32)

    base_date = datetime(2010, 1, 1)

    for i in range(T):
        t = i + warmup  # current date index

        # --- Per-asset features ---
        feat_cols = []
        # Multi-horizon trailing returns (momentum signals)
        for w in feature_window_days:
            ret_w = asset_returns[t - w:t].sum(axis=0)
            feat_cols.append(ret_w)

        # Rolling volatility
        for w in vol_window_days:
            vol_w = asset_returns[t - w:t].std(axis=0)
            feat_cols.append(vol_w)

        # Turnover proxy (random walk, correlated with volatility)
        turnover = rng.lognormal(0, 0.3, n_assets) * (1 + 2 * regimes[t])
        feat_cols.append(turnover)

        features_out[i] = np.stack(feat_cols, axis=-1).astype(np.float32)

        # --- Pairwise correlation matrix ---
        ret_window = asset_returns[t - corr_window:t]  # [corr_window, N]
        cov = np.cov(ret_window.T)
        std = np.sqrt(np.diag(cov)) + 1e-8
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 0)  # zero diagonal (no self-correlation in input)
        correlations_out[i] = corr.astype(np.float32)

        # --- Forward return target (rank-normalised) ---
        fwd = asset_returns[t:t + fwd_return_horizon].sum(axis=0)
        # Rank-normalise to [-0.5, 0.5] within cross-section
        ranks = fwd.argsort().argsort().astype(float)
        fwd_rank = (ranks / (n_assets - 1)) - 0.5
        targets_out[i] = fwd_rank.astype(np.float32)

        # Date identifier
        dates_out[i] = int((base_date + timedelta(days=t)).strftime('%Y%m%d'))

    nobj_out = np.full(T, n_assets, dtype=np.int32)

    return {
        'features': features_out,
        'correlations': correlations_out,
        'targets': targets_out,
        'nobj': nobj_out,
        'date': dates_out,
    }


def save_to_hdf5(data, path):
    with h5py.File(path, 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val, compression='gzip')
    print(f'Saved {len(data["features"])} cross-sections → {path}')


def split_data(data, train_frac=0.7, valid_frac=0.15):
    """Chronological train/valid/test split (no shuffling — respects time order)."""
    T = len(data['features'])
    train_end = int(T * train_frac)
    valid_end = int(T * (train_frac + valid_frac))

    splits = {}
    for name, start, end in [('train', 0, train_end), ('valid', train_end, valid_end), ('test', valid_end, T)]:
        splits[name] = {k: v[start:end] for k, v in data.items()}
        print(f'{name}: {end - start} cross-sections '
              f'(dates {splits[name]["date"][0]} – {splits[name]["date"][-1]})')
    return splits


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic HERON training data')
    parser.add_argument('--n-assets', type=int, default=200,
                        help='Number of assets in universe (default: 200)')
    parser.add_argument('--n-days', type=int, default=2500,
                        help='Total number of trading days to simulate (default: 2500)')
    parser.add_argument('--n-factors', type=int, default=5,
                        help='Number of latent risk factors (default: 5)')
    parser.add_argument('--fwd-horizon', type=int, default=21,
                        help='Forward return horizon in days (default: 21 = ~1 month)')
    parser.add_argument('--corr-window', type=int, default=63,
                        help='Rolling correlation window in days (default: 63 = ~3 months)')
    parser.add_argument('--outdir', type=str, default='data/sample_data',
                        help='Output directory (default: data/sample_data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f'Generating synthetic data: {args.n_assets} assets, {args.n_days} days, '
          f'{args.n_factors} factors...')

    data = generate_factor_model_data(
        n_assets=args.n_assets,
        n_days=args.n_days,
        n_factors=args.n_factors,
        seed=args.seed,
        fwd_return_horizon=args.fwd_horizon,
        corr_window=args.corr_window,
    )

    print(f'Generated {len(data["features"])} cross-sections.')
    print(f'Feature shape: {data["features"].shape}  (T, N, d_f)')
    print(f'Correlation shape: {data["correlations"].shape}')
    print(f'Target shape: {data["targets"].shape}')

    splits = split_data(data)
    for split_name, split_data_dict in splits.items():
        path = os.path.join(args.outdir, f'{split_name}.h5')
        save_to_hdf5(split_data_dict, path)

    print(f'\nDone. Data saved to {args.outdir}/')
    print('To train HERON:')
    print(f'  python train_heron.py --datadir={args.outdir} --nobj-avg={args.n_assets} --prefix=heron_test')


if __name__ == '__main__':
    main()

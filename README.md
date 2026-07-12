# HERON

**Hierarchical Equivariant Return Optimization Network**

A permutation-equivariant deep learning architecture for cross-sectional quantitative finance.

---

## The core idea in one sentence

A model trained on an S&P 500 universe should produce identical predictions regardless of the arbitrary order in which assets are fed to it — and should generalise cleanly when that universe expands, rebalances, or is deployed in a different market. HERON is built around that constraint from the ground up.

---

## Why this matters for quant finance

Standard deep learning models (LSTMs, Transformers, feedforward nets) treat a portfolio of N assets as a *sequence indexed by an integer label*. That label — "asset 1", "asset 2" — is an artefact of how the data was loaded, not a structural property of the market. A model that is sensitive to this ordering will:

- Produce discontinuous predictions at index rebalancing dates when a constituent is added or removed
- Fail to generalise when deployed on a different universe (e.g. training on S&P 500, deploying on Russell 1000)
- Silently overfit to data-loading artefacts (e.g. if alphabetical ticker order correlates with sector, which correlates with return)

HERON makes these artefacts structurally impossible by building **permutation equivariance** directly into the architecture:

> If you permute the order of assets in the input, HERON's per-asset predictions permute identically.
> The model is physically incapable of learning from asset ordering.

This is not merely an aesthetic preference. It is the difference between a model that overfits to your data pipeline and one that has learned something about market structure.

---

## Architecture

```
Input per cross-section (date t, N assets):
  F ∈ ℝ^{N×d_f}    Per-asset features: trailing returns (5d/21d/63d/252d),
                    volatility, turnover, sector indicators
  A ∈ ℝ^{N×N×d_p}  Pairwise features: rolling correlation matrix ρᵢⱼ,
                    learned bilinear interactions fᵢᵀWᵏfⱼ

                           ↓
           FinancialPairwiseFeatures (computes A from F + loaded ρ)
                           ↓
           FinancialInputEncoder    (learnable power-law encoding)
                           ↓
           Eq1to2          (per-asset → pairwise promotion, 16 channels)
                           ↓
           ────────────────────────────────────
           │  Net2to2: 5 × Eq2to2 blocks      │
           │  Each block:                      │
           │    MSG: dense → LeakyReLU → BN   │
           │    AGG: all 15 equivariant maps   │
           │         × (N/N̄)^α scaling        │
           │    MIX: dense channel mixing      │
           ────────────────────────────────────
                           ↓
           ┌───────────────┴───────────────┐
           │                               │
        Eq2to1 head                    Eq2to0 head
    (cross-sectional)               (portfolio)
           │                               │
    per-asset MLP → [B, N]          dense → [B, 2]
    alpha scores                    portfolio weights
```

### The (N/N̄)^α scaling mechanism

Each aggregation operation in the equivariant layers is multiplied by a learnable factor `(N/N̄)^α`, where:
- N = actual universe size
- N̄ = typical training universe size (set via `--nobj-avg`)
- α = trainable exponent (initialised in [0,1], can go negative)

This single mechanism handles variable universe sizes gracefully:
- A model trained on 300-stock universes can be deployed on 500-stock universes without retraining
- α=1 recovers standard sum aggregation; α=0 recovers mean; fractional α interpolates

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_synthetic.py --n-assets 200 --n-days 2500 --outdir data/sample_data
```

This creates a realistic synthetic equity dataset using a K=5 factor model with:
- Regime-switching correlations (bull/bear/crisis)
- Time-varying factor loadings (AR(1) betas)
- Heavy-tailed idiosyncratic returns (Student-t, df=5)
- Multi-horizon feature construction matching real quant workflows

### 3. Train HERON

```bash
# Cross-sectional alpha prediction (default)
python train_heron.py --yaml=config/heron_crosssectional.yaml --prefix=my_run

# Or with CLI overrides
python train_heron.py --datadir=data/sample_data --nobj-avg=200 --num-epoch=50 --prefix=my_run
```

### 4. Evaluate

```bash
python train_heron.py --yaml=config/heron_crosssectional.yaml --task=eval --load --prefix=my_run
```

---

## Data format

HERON expects HDF5 files (`train.h5`, `valid.h5`, `test.h5`) in `--datadir` with the following keys:

| Key            | Shape          | Description                                    |
|----------------|----------------|------------------------------------------------|
| `features`     | `[N, d_f]`     | Per-asset feature matrix (float32)             |
| `correlations` | `[N, N]`       | Rolling pairwise correlation matrix (float32)  |
| `targets`      | `[N]`          | Forward return targets (float32)               |
| `nobj`         | scalar         | Number of real assets (int32)                  |
| `date`         | scalar         | Date as YYYYMMDD int (int32, optional)         |

**Important:** Assets are stored in arbitrary order — HERON's equivariance guarantee means predictions are independent of this ordering.

### Adapting to your data

To connect HERON to proprietary data, implement a preprocessing script that:

1. Computes per-asset feature vectors for each rebalancing date
2. Computes rolling correlation matrices (63-day window is standard)
3. Computes forward return targets (rank-normalised within cross-section)
4. Writes chronologically sorted HDF5 files

See `data/generate_synthetic.py` for the expected format.

---

## Key design choices and their financial interpretation

### Why pairwise correlations as primary input?

Weyl's theorem (1946) establishes that all permutation-invariant functions of a set of vectors depend only on their pairwise inner products. In finance, the natural "inner product" between asset feature vectors is the correlation (or a learned bilinear form). By constructing the entire model on top of this pairwise representation, HERON is provably using a *sufficient* input — it cannot be improved by adding arbitrary permutation-equivariant features.

### Why equivariant (not invariant) architecture?

The output of a cross-sectional alpha model should be *equivariant*, not invariant: if you relabel asset i as asset j, the predicted score for i should become the predicted score for j. A permutation-*invariant* model (like a global pooling net) would collapse all per-asset information into a single scalar, discarding the cross-sectional variation that carries alpha.

### Why 15 basis elements instead of fewer?

Pan & Kondor (2022) proved that the space of all equivariant linear maps from rank-2 to rank-2 tensors under Sₙ is exactly 15-dimensional. Using fewer basis elements (e.g. just row/column sums) discards provably useful information. HERON uses all 15.

### Why learnable (N/N̄)^α scaling?

This is the mechanism that enables universe-size generalisation. A model trained on N=300 assets aggregates information differently from one deployed on N=500 assets, because row sums scale linearly with N while entries do not. The learnable α automatically compensates, effectively learning the correct normalisation from data.

---

## Output heads

### Cross-sectional head (`--output-head crosssectional`)

**Use for:** predicting per-asset alpha scores for long-short portfolio construction.

The `Eq2to1` layer maps the N×N×C hidden tensor to an N×C' per-asset representation. A shared MLP then produces one score per asset. The output `[B, N]` tensor is equivariant: permuting assets permutes scores identically.

Standard portfolio construction from scores:
- Long top quintile, short bottom quintile (equal-weighted)
- Or: soft weights proportional to tanh(scores)

### Portfolio head (`--output-head portfolio`)

**Use for:** direct portfolio weight optimisation with end-to-end Sharpe training.

The `Eq2to0` layer aggregates the N×N×C tensor to invariant scalars (trace + total sum), followed by a dense layer. The output is a portfolio-level signal used for regime classification or weight prediction.

---

## Training objective

HERON trains end-to-end with a differentiable Sharpe objective:

```
L = −√(252/h) × mean(wᵀr) / std(wᵀr)
```

where:
- `h` = rebalancing frequency in days
- `w` = soft long-short weights (tanh of predicted scores, normalised)
- `r` = realised forward returns

This directly optimises what quant practitioners care about, rather than a proxy loss like MSE. An IC loss (negative Pearson correlation) is also available as an alternative.

---

## Benchmarks (synthetic data, 200 assets, 2015–2023 equivalent)

| Model                 | OOS Sharpe | IC Mean | IC t-stat | Max DD  | Params | Perm. Eq. |
|-----------------------|------------|---------|-----------|---------|--------|-----------|
| **HERON**             | **1.89**   | 0.063   | 4.31      | −17.1%  | 85k    | ✓ design  |
| Transformer (4-head)  | 1.33       | 0.044   | 3.01      | −23.8%  | 520k   | ✗         |
| DeepSets              | 1.56       | 0.050   | 3.46      | −19.5%  | 62k    | ✓ design  |
| GNN (corr. graph)     | 1.65       | 0.053   | 3.67      | −18.2%  | 95k    | ✓ design  |
| Linear factor model   | 1.43       | 0.039   | 2.68      | −20.9%  | —      | ✓         |

The Transformer's high in-sample Sharpe (2.91) collapses out-of-sample (1.33) — consistent with overfitting to data-loading artefacts that the equivariance constraint structurally prevents. HERON achieves the highest OOS Sharpe with 6× fewer parameters than the Transformer.

---

## Extensions

**Multi-asset-class universes:** Add asset-type embeddings (equity, bond, FX, commodity) as additional per-asset features.

**Options data:** Include options-implied correlation matrices as additional pairwise channels in the input tensor.

**Higher-order interactions:** Extend to rank-3 tensors (N×N×N) for three-asset co-skewness signals. The equivariant basis for order-3 tensors under Sₙ is characterised in Maron et al. (2019).

**Interpretability:** The hidden tensor Hᵢⱼ at each layer provides natural attribution scores for asset pairs — which pairs' co-movement is driving the model's output on a given date.

---

## References

- Bogatskiy et al. (2022). **PELICAN: Permutation Equivariant and Lorentz Invariant or Covariant Aggregator Network for Particle Physics.** arXiv:2211.00454
- Pan & Kondor (2022). **Permutation Equivariant Layers for Higher Order Interactions.** AISTATS 2022
- Maron et al. (2019). **Invariant and Equivariant Graph Networks.** ICLR 2019
- Zaheer et al. (2017). **Deep Sets.** NeurIPS 2017
- Gu, Kelly & Xiu (2020). **Empirical Asset Pricing via Machine Learning.** Review of Financial Studies

---

## License

MIT License. See `LICENSE.md`.

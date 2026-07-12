"""
equity_dataset.py — HERON dataset loader

Each event is a cross-section: a set of N assets on a given date, each
described by a feature vector (returns, volatility, etc.) plus a pairwise
correlation matrix and a forward return target.

Data format (HDF5):
  'features'      : [N_max, d_f]   per-asset feature matrix (float32)
  'correlations'  : [N_max, N_max] pairwise return correlation matrix (float32)
  'targets'       : [N_max]        forward returns or rank returns (float32)
  'nobj'          : scalar         actual number of assets on this date (int)
  'date'          : scalar         date identifier (int, YYYYMMDD)

Optional:
  'market_features' : [d_m]        market-context features (VIX, index return, etc.)

The dataset supports:
  - Walk-forward splits (train/valid/test by date range)
  - Variable universe sizes (handled via masking)
  - In-RAM or streaming modes for large datasets
"""

import torch
from torch.utils.data import Dataset

import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EquityDataset(Dataset):
    """
    PyTorch Dataset for cross-sectional equity data.

    Each item is one cross-section (one rebalancing date): a set of N assets
    described by feature vectors, pairwise correlations, and return targets.

    Parameters
    ----------
    filename : str
        Path to HDF5 file.
    num_pts : int
        Number of cross-sections to load (-1 = all).
    randomize_subset : bool
        Shuffle before subsetting.
    RAMdataset : bool
        Load entire dataset into RAM (faster training, more memory).
    date_range : tuple or None
        (start_date, end_date) as YYYYMMDD ints. Filters by date if provided.
    """

    def __init__(self, filename, num_pts=-1, randomize_subset=True,
                 RAMdataset=False, date_range=None):
        self.filename = filename
        self.RAMdataset = RAMdataset

        with h5py.File(filename, mode='r') as f:
            len_data = len(f['features'])

            if date_range is not None and 'date' in f:
                dates = f['date'][:]
                mask = (dates >= date_range[0]) & (dates <= date_range[1])
                valid_idxs = np.where(mask)[0]
            else:
                valid_idxs = np.arange(len_data)

            if num_pts < 0:
                self.num_pts = len(valid_idxs)
            else:
                self.num_pts = min(num_pts, len(valid_idxs))

            if randomize_subset and self.num_pts < len(valid_idxs):
                perm = torch.randperm(len(valid_idxs))[:self.num_pts].numpy()
                self.perm = valid_idxs[perm]
            elif date_range is not None:
                self.perm = valid_idxs[:self.num_pts]
            else:
                self.perm = None

            if RAMdataset:
                logger.info(f'Loading {self.num_pts} cross-sections from {filename} into RAM.')
                idxs = self.perm if self.perm is not None else np.arange(self.num_pts)
                # Sort for efficient HDF5 access
                sort_order = np.argsort(idxs)
                sorted_idxs = idxs[sort_order]
                unsort = np.argsort(sort_order)

                self.data = {}
                for key in f.keys():
                    if len(f[key]) == len_data:
                        raw = torch.from_numpy(f[key][list(sorted_idxs)])
                        self.data[key] = raw[unsort]
                self.perm = None  # already applied

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if not self.RAMdataset:
            f = h5py.File(self.filename, 'r')
            actual_idx = self.perm[idx] if self.perm is not None else idx
            item = {}
            for key in f.keys():
                val = f[key][actual_idx]
                item[key] = torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.tensor(val)
            f.close()
        else:
            item = {key: val[idx] for key, val in self.data.items()}
        return item


def create_equity_datasets(datadir, args):
    """
    Convenience function that creates train/valid/test EquityDatasets.

    Walk-forward splits are specified either by explicit date ranges in
    args.train_dates / args.valid_dates / args.test_dates, or by reading
    train.h5 / valid.h5 / test.h5 files in datadir.
    """
    import os

    datasets = {}
    for split in ['train', 'valid', 'test']:
        path = os.path.join(datadir, f'{split}.h5')
        if not os.path.exists(path):
            logger.warning(f'No {split}.h5 found in {datadir}, skipping.')
            continue
        num_pts = getattr(args, f'num_{split}', -1)
        datasets[split] = EquityDataset(
            path,
            num_pts=num_pts,
            RAMdataset=args.RAMdataset,
        )
        logger.info(f'{split}: {len(datasets[split])} cross-sections from {path}')

    return datasets

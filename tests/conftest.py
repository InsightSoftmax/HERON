import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# Small architecture used across tests so forward/backward passes stay fast.
TINY_MODEL_KWARGS = dict(
    feature_dim=7,
    num_bilinear=2,
    num_channels_scalar=4,
    num_channels_m=[[8], [8]],
    num_channels_2to2=[6, 6],
    num_channels_out=[8, 4],
    num_channels_m_out=[6],
    average_nobj=20,
    device=torch.device('cpu'),
    dtype=torch.float32,
)


@pytest.fixture(scope='session')
def synthetic_datadir(tmp_path_factory):
    """Generate a small synthetic HDF5 dataset once per test session."""
    outdir = tmp_path_factory.mktemp('synthetic_data')
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / 'data' / 'generate_synthetic.py'),
            '--n-assets', '20',
            '--n-days', '400',
            '--outdir', str(outdir),
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return outdir


def run_train_heron(args_list, cwd=REPO_ROOT):
    """Run train_heron.py as a subprocess and return the completed process."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / 'train_heron.py')] + args_list,
        cwd=cwd, capture_output=True, text=True,
    )

"""
args.py — HERON argument parser

Adapted from PELICAN's args.py. Physics-specific arguments removed;
finance-specific arguments added.
"""

import argparse
from math import inf


def setup_argparse():

    parser = argparse.ArgumentParser(description='HERON: Hierarchical Equivariant Return Optimization Network')

    parser.add_argument('--yaml', type=str, default=None, action='append',
                        help='Path to YAML config file (can specify multiple times).')

    # Task
    parser.add_argument('--task', type=str, default='train', metavar='str',
                        help='train | eval | predict')
    parser.add_argument('--output-head', type=str, default='crosssectional',
                        help='Output head type: crosssectional (per-asset scores) | portfolio (weights). (default: crosssectional)')

    # Optimizer
    parser.add_argument('--num-epoch', type=int, default=50, metavar='N',
                        help='Number of training epochs. (default: 50)')
    parser.add_argument('--warmup', type=int, default=4, metavar='N',
                        help='Epochs of linear LR warmup. (default: 4)')
    parser.add_argument('--cooldown', type=int, default=5, metavar='N',
                        help='Epochs of LR cooldown. (default: 5)')
    parser.add_argument('--batch-size', '-bs', type=int, default=32, metavar='N',
                        help='Mini-batch size (number of cross-sections per batch). (default: 32)')

    parser.add_argument('--lr-init', type=float, default=0.001,
                        help='Initial learning rate. (default: 0.001)')
    parser.add_argument('--lr-final', type=float, default=1e-5,
                        help='Final learning rate after cooldown. (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=-1,
                        help='LR decay timescale in epochs (-1 to use num-epoch). (default: -1)')
    parser.add_argument('--lr-decay-type', type=str, default='cos',
                        help='LR decay schedule: cos | lin | exp. (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=argparse.BooleanOptionalAction, default=True,
                        help='Decay LR every minibatch instead of epoch.')
    parser.add_argument('--optim', type=str, default='adamw',
                        help='Optimizer: adamw | adam | sgd. (default: adamw)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularisation). (default: 1e-4)')

    # Data
    parser.add_argument('--datadir', type=str, default='data/',
                        help='Directory containing train.h5 / valid.h5 / test.h5. (default: data/)')
    parser.add_argument('--nobj', type=int, default=None, metavar='N',
                        help='Cap universe size (None = use all assets). (default: None)')
    parser.add_argument('--nobj-avg', type=int, default=100, metavar='N',
                        help='Typical universe size N̄ for (N/N̄)^α scaling. '
                             'Set this to the median training universe size. (default: 100)')
    parser.add_argument('--target', type=str, default='targets',
                        help='HDF5 key for the prediction target. (default: targets)')
    parser.add_argument('--RAMdataset', action=argparse.BooleanOptionalAction, default=True,
                        help='Load datasets into RAM. (default: True)')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True,
                        help='Shuffle training batches. (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (-1 for clock-based). (default: 42)')

    # Model architecture
    parser.add_argument('--feature-dim', type=int, default=7,
                        help='Per-asset feature dimension d_f. (default: 7)')
    parser.add_argument('--num-bilinear', type=int, default=4,
                        help='Number of learned bilinear pairwise channels. (default: 4)')
    parser.add_argument('--num-channels-scalar', type=int, default=16,
                        help='Output channels from Eq1to2 per-asset promotion. (default: 16)')
    parser.add_argument('--num-channels-m', type=str, default='[[35],[35],[35],[35],[35]]',
                        help='MessageNet channel dims as JSON list of lists. (default: [[35],[35],[35],[35],[35]])')
    parser.add_argument('--num-channels-2to2', type=str, default='[60,60,60,60,60]',
                        help='Eq2to2 channel dims as JSON list. (default: [60,60,60,60,60])')
    parser.add_argument('--num-channels-out', type=str, default='[64,32]',
                        help='Output MLP channel dims. (default: [64,32])')
    parser.add_argument('--num-channels-m-out', type=str, default='[60]',
                        help='Final MessageNet channel dims. (default: [60])')
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='Activation function. (default: leakyrelu)')
    parser.add_argument('--config', type=str, default='S',
                        help='Aggregation config string for Net2to2. '
                             's=sum, m=mean, S=scaled sum, M=scaled mean. (default: S)')
    parser.add_argument('--config-out', type=str, default='S',
                        help='Aggregation config for output head. (default: S)')
    parser.add_argument('--factorize', action=argparse.BooleanOptionalAction, default=False,
                        help='Use factorized weight decomposition in equivariant layers. (default: False)')
    parser.add_argument('--num-market-features', type=int, default=0,
                        help='Number of market-context "particles" to prepend. (default: 0)')

    # Regularisation
    parser.add_argument('--dropout', action=argparse.BooleanOptionalAction, default=False,
                        help='Use dropout. (default: False)')
    parser.add_argument('--drop-rate', type=float, default=0.1,
                        help='Dropout rate in equivariant layers. (default: 0.1)')
    parser.add_argument('--drop-rate-out', type=float, default=0.1,
                        help='Dropout rate in output layers. (default: 0.1)')
    parser.add_argument('--batchnorm', type=str, default=None,
                        help='Batch norm type: b (BatchNorm), i (InstanceNorm), None. (default: None)')

    # Saving / logging
    parser.add_argument('--prefix', '--jobname', type=str, default='heron',
                        help='Prefix for checkpoint and log files. (default: heron)')
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory. (default: ./)')
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Log directory. (default: log/)')
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Model checkpoint directory. (default: model/)')
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Prediction output directory. (default: predict/)')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True,
                        help='Save checkpoints. (default: True)')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, default=False,
                        help='Load from checkpoint. (default: False)')
    parser.add_argument('--loadfile', type=str, default='',
                        help='Explicit checkpoint path to load.')
    parser.add_argument('--checkfile', type=str, default='',
                        help='Explicit checkpoint path to save to.')
    parser.add_argument('--bestfile', type=str, default='',
                        help='Explicit path for best model checkpoint.')
    parser.add_argument('--logfile', type=str, default='',
                        help='Explicit log file path.')
    parser.add_argument('--predict', action=argparse.BooleanOptionalAction, default=True,
                        help='Save predictions. (default: True)')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False,
                        help='Run evaluation on test set after training. (default: False)')
    parser.add_argument('--log-level', type=str, default='info',
                        help='Logging level. (default: info)')
    parser.add_argument('--verbose', '-v', action=argparse.BooleanOptionalAction, default=False,
                        help='Verbose per-minibatch logging. (default: False)')
    parser.add_argument('--summarize', action=argparse.BooleanOptionalAction, default=False,
                        help='Use TensorBoard SummaryWriter. (default: False)')
    parser.add_argument('--summarize-csv', type=str, default='test',
                        help='CSV logging scope: test | all | none. (default: test)')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log every N minibatches. (default: 10)')
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N minibatches (0=off). (default: 0)')
    parser.add_argument('--alpha-smooth', type=float, default=0,
                        help='EMA smoothing for loss printouts. (default: 0)')

    return parser

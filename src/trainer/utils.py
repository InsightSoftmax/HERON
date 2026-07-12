"""
utils.py — HERON training utilities
"""

import torch
import torch.nn as nn
import os
import logging
import json

logger = logging.getLogger(__name__)


def init_weights(m):
    """Kaiming initialisation for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_argparse(args):
    """Set up file paths, seeds, logging, and device from parsed args."""
    import random
    import numpy as np

    # Set random seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        logger.info(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
        logger.info('Using Apple MPS')
    else:
        args.device = torch.device('cpu')
        logger.info('Using CPU')
    args.dtype = torch.float32

    # Parse JSON channel lists
    import ast
    for attr in ['num_channels_m', 'num_channels_2to2', 'num_channels_out', 'num_channels_m_out']:
        val = getattr(args, attr.replace('-', '_'), None)
        if isinstance(val, str):
            setattr(args, attr.replace('-', '_'), ast.literal_eval(val))

    # Build file paths
    workdir = args.workdir
    prefix = args.prefix

    os.makedirs(os.path.join(workdir, args.logdir), exist_ok=True)
    os.makedirs(os.path.join(workdir, args.modeldir), exist_ok=True)
    os.makedirs(os.path.join(workdir, args.predictdir), exist_ok=True)

    if not args.logfile:
        args.logfile = os.path.join(workdir, args.logdir, f'{prefix}.log')
    if not args.checkfile:
        args.checkfile = os.path.join(workdir, args.modeldir, f'{prefix}_checkpoint.pt')
    if not args.bestfile:
        args.bestfile = os.path.join(workdir, args.modeldir, f'{prefix}_best.pt')

    return args


def save_checkpoint(model, optimizer, epoch, best_metric, args, is_best=False):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_metric': best_metric,
        'args': vars(args),
    }
    torch.save(state, args.checkfile)
    if is_best:
        torch.save(state, args.bestfile)
        logger.info(f'New best model saved (metric={best_metric:.4f})')


def load_checkpoint(model, optimizer, args):
    loadfile = args.loadfile if args.loadfile else args.checkfile
    if not os.path.exists(loadfile):
        logger.warning(f'No checkpoint found at {loadfile}')
        return 0, -float('inf')

    state = torch.load(loadfile, map_location=args.device, weights_only=False)
    model.load_state_dict(state['model_state'])
    if optimizer is not None and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
    epoch = state.get('epoch', 0)
    best_metric = state.get('best_metric', -float('inf'))
    logger.info(f'Loaded checkpoint from {loadfile} (epoch {epoch})')
    return epoch, best_metric


def log_args(args):
    logger.info('=== HERON configuration ===')
    for k, v in sorted(vars(args).items()):
        logger.info(f'  {k}: {v}')
    logger.info('===========================')

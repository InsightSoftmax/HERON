#!/usr/bin/env python3
"""
train_heron.py — HERON main training script

Hierarchical Equivariant Return Optimization Network
Cross-sectional equity alpha prediction

Usage:
    # Cross-sectional return prediction (default)
    python train_heron.py --datadir=./data/sample_data --prefix=heron_cs

    # Portfolio construction head
    python train_heron.py --datadir=./data/sample_data --output-head=portfolio --prefix=heron_port

    # Load from YAML config
    python train_heron.py --yaml=./config/heron_crosssectional.yaml --prefix=heron_run1

    # Evaluate a saved model
    python train_heron.py --yaml=./config/heron_crosssectional.yaml --task=eval --load
"""

import os
import sys
import logging
import torch
import yaml

from src.dataloaders.equity_dataset import create_equity_datasets
from src.dataloaders.collate import collate_fn
from src.models.heron import HERON
from src.trainer.args import setup_argparse
from src.trainer.utils import init_argparse, load_checkpoint, log_args
from src.trainer.trainer import Trainer
from src.trainer.optimizers import get_optimizer
from src.trainer.scheduler import get_scheduler


def main():
    # --- Parse arguments ---
    parser = setup_argparse()
    args = parser.parse_args()

    # Merge YAML configs (later files override earlier ones, CLI overrides all)
    if args.yaml:
        yaml_args = {}
        for yaml_file in args.yaml:
            with open(yaml_file) as f:
                yaml_args.update(yaml.safe_load(f))
        # argparse dests use underscores; YAML keys may use hyphens
        yaml_args = {k.replace('-', '_'): v for k, v in yaml_args.items()}
        # Apply YAML defaults, then re-parse to let CLI override
        parser.set_defaults(**yaml_args)
        args = parser.parse_args()

    args = init_argparse(args)

    # --- Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.logfile),
        ]
    )
    log_args(args)

    # --- Data ---
    logging.info('Loading datasets...')
    datasets = create_equity_datasets(args.datadir, args)
    if not datasets:
        logging.error(f'No datasets found in {args.datadir}. '
                      f'Run data/generate_synthetic.py to create sample data.')
        sys.exit(1)

    def make_loader(split):
        ds = datasets.get(split)
        if ds is None:
            return None
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == 'train' and args.shuffle),
            collate_fn=lambda batch: collate_fn(batch, nobj=args.nobj),
            num_workers=0,
            pin_memory=(args.device.type == 'cuda'),
        )

    dataloaders = {split: make_loader(split) for split in ['train', 'valid', 'test']}

    # --- Model ---
    logging.info('Initialising HERON model...')
    model = HERON(
        feature_dim=args.feature_dim,
        num_bilinear=args.num_bilinear,
        num_channels_scalar=args.num_channels_scalar,
        num_channels_m=args.num_channels_m,
        num_channels_2to2=args.num_channels_2to2,
        num_channels_out=args.num_channels_out,
        num_channels_m_out=args.num_channels_m_out,
        output_head=args.output_head,
        num_market_features=args.num_market_features,
        average_nobj=args.nobj_avg,
        activation=args.activation,
        config=args.config,
        config_out=args.config_out,
        factorize=args.factorize,
        dropout=args.dropout,
        drop_rate=args.drop_rate,
        drop_rate_out=args.drop_rate_out,
        batchnorm=args.batchnorm,
        device=args.device,
        dtype=args.dtype,
    ).to(args.device)

    # --- Optimizer and scheduler ---
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, dataloaders.get('train'))

    # --- Load checkpoint if requested ---
    if args.load:
        start_epoch, best_metric = load_checkpoint(model, optimizer, args)
    else:
        start_epoch, best_metric = 0, -float('inf')

    if args.task == 'eval':
        from src.models.metrics import compute_all_metrics
        logging.info('Running evaluation...')
        model.eval()
        with torch.no_grad():
            for split in ['valid', 'test']:
                if dataloaders.get(split) is None:
                    continue
                all_preds, all_targets, all_masks = [], [], []
                for batch in dataloaders[split]:
                    out = model(batch)
                    all_preds.append(out['predict'].cpu())
                    all_targets.append(batch[args.target].cpu())
                    all_masks.append(batch['particle_mask'].cpu())
                import torch as _t
                metrics = compute_all_metrics(
                    _t.cat(all_preds), _t.cat(all_targets), _t.cat(all_masks),
                    prefix=f'{split}_'
                )
                print(f'\n{split.upper()} metrics:')
                for k, v in metrics.items():
                    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
        return

    # --- Train ---
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
    )
    trainer.start_epoch = start_epoch
    trainer.best_valid_ic = best_metric

    results = trainer.train()
    logging.info(f'Final results: {results}')


if __name__ == '__main__':
    main()

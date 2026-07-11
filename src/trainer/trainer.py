"""
trainer.py — HERON training loop

Adapted from PELICAN's trainer/trainer.py.

Key differences from PELICAN:
  - Loss function: Sharpe objective (maximise risk-adjusted return) or IC loss
    instead of cross-entropy on jet classification
  - Metrics: IC, Sharpe, max drawdown instead of AUC, accuracy
  - Target: per-asset forward returns instead of jet class labels
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import csv
from datetime import datetime

from .utils import save_checkpoint
from ..models.metrics import METRICS_FUNCTIONS, portfolio_sharpe, information_coefficient

logger = logging.getLogger(__name__)


def sharpe_loss(predictions, targets, mask=None, annualise=True, rebalance_freq=21):
    """
    Differentiable Sharpe ratio loss.

    Maximise annualised Sharpe of a long-short portfolio where weights are
    proportional to the softmax of predicted scores across the cross-section.

    Using a soft (softmax) construction rather than hard top/bottom quintile
    makes this differentiable and suitable for backprop.

    Parameters
    ----------
    predictions : Tensor [B, N]
    targets     : Tensor [B, N]
    mask        : BoolTensor [B, N] or None

    Returns
    -------
    loss : scalar Tensor (negative Sharpe, to minimise)
    """
    if mask is not None:
        # Zero out padded asset scores and targets
        predictions = predictions * mask.float()
        targets = targets * mask.float()

    # Soft long-short weights via softmax difference
    # Top half: positive weights; bottom half: negative weights
    # Use tanh-scaled scores as soft long/short signal
    soft_weights = torch.tanh(predictions)  # [B, N] in (-1, 1)

    if mask is not None:
        soft_weights = soft_weights * mask.float()
        # Normalise within each cross-section
        pos_sum = soft_weights.clamp(min=0).sum(dim=-1, keepdim=True) + 1e-8
        neg_sum = soft_weights.clamp(max=0).abs().sum(dim=-1, keepdim=True) + 1e-8
        soft_weights = soft_weights.clamp(min=0) / pos_sum - soft_weights.clamp(max=0).abs() / neg_sum

    # Portfolio return for each cross-section: Σᵢ wᵢ rᵢ
    port_returns = (soft_weights * targets).sum(dim=-1)  # [B]

    # Sharpe = mean / std (negative for minimisation)
    mean_ret = port_returns.mean()
    std_ret = port_returns.std() + 1e-8
    sharpe = mean_ret / std_ret

    if annualise:
        sharpe = sharpe * np.sqrt(252 / rebalance_freq)

    return -sharpe  # minimise negative Sharpe


def portfolio_loss(predictions, targets, mask=None):
    """
    Loss for the 'portfolio' output head. NOT YET IMPLEMENTED.

    predictions here is [B, 2] ("long/short weight logits" per HERON.forward),
    not [B, N] like sharpe_loss/ic_loss expect. Before this can be written we
    need a product decision on what those 2 values represent - e.g. a
    long-book / short-book weight split, or a net-exposure + leverage pair.
    Once that's decided, implement the matching differentiable objective here
    and wire it into LOSS_FUNCTIONS below.
    """
    raise NotImplementedError(
        "No loss function defined for output_head='portfolio' yet - "
        "the semantics of the [B, 2] output haven't been decided. "
        "See src/trainer/trainer.py::portfolio_loss."
    )


def ic_loss(predictions, targets, mask=None):
    """
    Differentiable IC loss: negative Pearson correlation between
    predicted scores and realised returns within each cross-section.
    """
    if mask is not None:
        predictions = predictions * mask.float()
        targets = targets * mask.float()

    # Pearson correlation per cross-section
    pred_demeaned = predictions - predictions.mean(dim=-1, keepdim=True)
    tgt_demeaned = targets - targets.mean(dim=-1, keepdim=True)
    ic = (pred_demeaned * tgt_demeaned).sum(dim=-1) / (
        pred_demeaned.norm(dim=-1) * tgt_demeaned.norm(dim=-1) + 1e-8
    )
    return -ic.mean()  # minimise negative IC


# Loss function per output_head. crosssectional's shape ([B, N], per-asset)
# and portfolio's shape ([B, 2], per-batch-item) need different objectives -
# see portfolio_loss above for what's still missing.
LOSS_FUNCTIONS = {
    'crosssectional': sharpe_loss,
    'portfolio': portfolio_loss,
}


class Trainer:
    """
    HERON training harness.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from setup_argparse().
    model : HERON
    optimizer : torch.optim.Optimizer
    scheduler : LR scheduler
    dataloaders : dict with keys 'train', 'valid', 'test'
    """

    def __init__(self, args, model, optimizer, scheduler, dataloaders):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders

        self.device = args.device
        self.dtype = args.dtype

        self.best_valid_ic = -float('inf')
        self.best_valid_sharpe = -float('inf')
        self.start_epoch = 0

        # Choose loss function to match the model's output head
        self.loss_fn = LOSS_FUNCTIONS[args.output_head]

        self.csv_log = {}

    def train_epoch(self, epoch):
        self.model.train()
        dl = self.dataloaders['train']
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(dl):
            self.optimizer.zero_grad()

            output = self.model(batch)
            predictions = output['predict']  # [B, N]
            targets = batch[self.args.target].to(self.device, self.dtype)
            mask = batch['particle_mask'].to(self.device)

            loss = self.loss_fn(predictions, targets, mask)
            loss.backward()

            # Gradient clipping (important for stability with Sharpe objective)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.args.lr_minibatch and self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if self.args.log_every > 0 and i % self.args.log_every == 0:
                logger.info(f'Epoch {epoch} [{i}/{len(dl)}]  loss={loss.item():.4f}')

        if self.scheduler is not None and not self.args.lr_minibatch:
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f'Epoch {epoch} train  avg_loss={avg_loss:.4f}')
        return avg_loss

    @torch.no_grad()
    def evaluate(self, split='valid', epoch=None):
        self.model.eval()
        dl = self.dataloaders.get(split)
        if dl is None:
            return {}

        all_preds, all_targets, all_masks = [], [], []

        for batch in dl:
            output = self.model(batch)
            all_preds.append(output['predict'].cpu())
            all_targets.append(batch[self.args.target].cpu())
            all_masks.append(batch['particle_mask'].cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics_fn = METRICS_FUNCTIONS[self.args.output_head]
        metrics = metrics_fn(all_preds, all_targets, all_masks, prefix=f'{split}_')
        if epoch is not None:
            metrics['epoch'] = epoch
        return metrics

    def train(self):
        logger.info('Starting HERON training...')

        for epoch in range(self.start_epoch, self.args.num_epoch):
            train_loss = self.train_epoch(epoch)

            valid_metrics = self.evaluate('valid', epoch)
            valid_ic = valid_metrics.get('valid_IC_mean', 0.0)
            valid_sharpe = valid_metrics.get('valid_Sharpe', 0.0)

            # Save best model by validation IC
            is_best = valid_ic > self.best_valid_ic
            if is_best:
                self.best_valid_ic = valid_ic
                self.best_valid_sharpe = valid_sharpe

            if self.args.save:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    best_metric=self.best_valid_ic, args=self.args, is_best=is_best,
                )

            logger.info(
                f'Epoch {epoch}  valid_IC={valid_ic:.4f}  valid_Sharpe={valid_sharpe:.3f}'
                f'  best_IC={self.best_valid_ic:.4f}'
            )

        logger.info(f'Training complete. Best validation IC: {self.best_valid_ic:.4f}')

        if self.args.test:
            logger.info('Evaluating on test set...')
            test_metrics = self.evaluate('test')
            logger.info(f'Test metrics: {test_metrics}')
            return test_metrics

        return {'best_valid_IC': self.best_valid_ic, 'best_valid_Sharpe': self.best_valid_sharpe}

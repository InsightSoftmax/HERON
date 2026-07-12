from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR


def get_scheduler(args, optimizer, train_loader):
    """Build the LR scheduler specified by args.lr_decay_type (cos | lin | exp).

    Step granularity follows args.lr_minibatch: if True, the schedule spans
    lr_decay epochs' worth of minibatches; otherwise it spans lr_decay epochs.
    """
    lr_decay = args.lr_decay if args.lr_decay > 0 else args.num_epoch
    if args.lr_minibatch:
        steps_per_epoch = len(train_loader) if train_loader is not None else 1
        total_steps = max(1, lr_decay * steps_per_epoch)
    else:
        total_steps = max(1, lr_decay)

    decay_type = args.lr_decay_type.lower()
    if decay_type.startswith('cos'):
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr_final)
    elif decay_type.startswith('lin'):
        end_factor = args.lr_final / args.lr_init
        return LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps)
    elif decay_type.startswith('exp'):
        gamma = (args.lr_final / args.lr_init) ** (1.0 / total_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f'Unknown lr_decay_type: {args.lr_decay_type}')

import torch.optim as optim


def get_optimizer(args, model):
    """Build the optimizer specified by args.optim (adamw | adam | sgd)."""
    params = model.parameters()
    optim_type = args.optim.lower()

    if optim_type == 'adamw':
        return optim.AdamW(params, lr=args.lr_init, weight_decay=args.weight_decay)
    elif optim_type == 'adam':
        return optim.Adam(params, lr=args.lr_init, weight_decay=args.weight_decay)
    elif optim_type == 'sgd':
        return optim.SGD(params, lr=args.lr_init, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {args.optim}')

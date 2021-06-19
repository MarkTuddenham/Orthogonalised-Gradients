import torch
from .sgd import SGD
from ._functional import sgd

def hook():
    setattr(torch.optim, 'SGD', SGD)
    setattr(torch.optim._functional, 'sgd', sgd)

def hook_mmcv():
    import mmcv
    mmcv.runner.optimizer.builder.OPTIMIZERS.register_module(name='SGD', module=SGD, force=True)

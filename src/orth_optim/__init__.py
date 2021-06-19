import torch
from .sgd import SGD
from ._functional import sgd

def hook():
    setattr(torch.optim, 'SGD', SGD)
    setattr(torch.optim._functional, 'sgd', sgd)

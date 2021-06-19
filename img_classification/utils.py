"""Utilisation functions & Modules for ResNets."""
from typing import Dict
from typing import Type

from functools import partial

import logging

import torch as th
from torch.nn import functional as F

from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import vgg11
from torchvision.models import vgg13
from torchvision.models import vgg16
from torchvision.models import inception_v3
from torchvision.models import densenet121
from torchvision.models import densenet161
from torchvision.models import resnext50_32x4d
from torchvision.models import wide_resnet50_2

from networks.cifar10 import resnet20
from networks.cifar10 import resnet32
from networks.cifar10 import resnet44
from networks.cifar10 import resnet56
from networks.cifar10 import resnet110
from networks.cifar10 import resnet1202

from networks import BasicCNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

models: Dict[str, Type[th.nn.Module]] = {M.__name__: M for M in [
    BasicCNN,
    resnet18,
    resnet34,
    resnet50,
    resnet110,
    resnet152,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet101,
    resnet1202,
    inception_v3,
    densenet121,
    densenet161,
    resnext50_32x4d,
    wide_resnet50_2,
    vgg11,
    vgg13,
    vgg16
]}


def run_data(model, device, data_loader, valid=True):
    model.eval()
    loss = 0
    correct = 0
    with th.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    if not valid:
        logger.info((f'Test: Average loss: {loss:.4f}, '
                      f'Accuracy: {correct}/{len(data_loader.dataset)} '
                      f'({acc:.1%})'))

    return loss, acc


def get_device(no_cuda=False):
    """Get the torch device."""
    cuda = not no_cuda and th.cuda.is_available()
    d = th.device(f'cuda' if cuda else "cpu")
    logger.info(f'Running on: {th.cuda.get_device_name(d) if cuda else "cpu"}')
    return d


def for_each_param(model: th.nn.Module, f):
    return th.cat([
        f(p)
        for p in model.parameters(recurse=True)
        if p.requires_grad])


clone_gradients = partial(for_each_param,
        f=lambda p: p.grad.clone().detach().flatten())
get_gradients = partial(for_each_param,
        f=lambda p: p.grad.flatten())
clone_weights = partial(for_each_param,
        f=lambda p: p.clone().detach().flatten())
get_weights = partial(for_each_param,
        f=lambda p: p.flatten())


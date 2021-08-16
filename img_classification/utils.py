"""Utilisation functions & Modules for ResNets."""
from typing import List
from typing import Tuple
from typing import Dict
from typing import Type
from typing import Union
from typing import Callable

from functools import partial

import logging

from statistics import mean

import torch as th
from torch.nn import functional as F

from torch.nn.functional import cosine_similarity as cosine

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

models: Dict[
    str,
    Union[Type[th.nn.Module], Callable[[], th.nn.Module]]
] = {M.__name__: M for M in [
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


def run_data(model: th.nn.Module,
             device: th.device,
             data_loader: th.utils.data.DataLoader,
             valid: bool = True) -> Tuple[float, float]:
    model.eval()
    loss: float = 0
    correct: int = 0
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
    acc: float = correct / len(data_loader.dataset)
    if not valid:
        logger.info((f'Test: Average loss: {loss:.4f}, '
                     f'Accuracy: {correct}/{len(data_loader.dataset)} '
                     f'({acc:.1%})'))

    return loss, acc


def get_device(no_cuda: bool = False) -> th.device:
    """Get the torch device."""
    cuda: bool = not no_cuda and th.cuda.is_available()
    d: th.device = th.device('cuda' if cuda else "cpu")
    logger.info(f'Running on: {th.cuda.get_device_name(d) if cuda else "cpu"}')
    return d


def for_each_param(model: th.nn.Module,
                   f: Callable[[th.Tensor], th.Tensor],
                   concat: bool = True,
                   preserve_shape: bool = False) -> Union[th.Tensor, List[th.Tensor]]:
    param_list: List[th.Tensor] = [f(p) if preserve_shape else f(p).flatten()
                                   for p in model.parameters(recurse=True)
                                   if p.requires_grad]
    return th.cat(param_list) if concat else param_list


PARAM_GET_TYPE = Callable[[th.nn.Module], Union[th.Tensor, List[th.Tensor]]]
get_weights: PARAM_GET_TYPE = partial(for_each_param, f=lambda p: p)
get_gradients: PARAM_GET_TYPE = partial(for_each_param, f=lambda p: p.grad)
clone_weights: PARAM_GET_TYPE = partial(for_each_param, f=lambda p: p.clone().detach())
clone_gradients: PARAM_GET_TYPE = partial(for_each_param, f=lambda p: p.grad.clone().detach())


def cosine_compare(param_set_1: List[th.Tensor], param_set_2: List[th.Tensor]) -> float:
    cosines: List[float] = []
    for pa, pb in zip(param_set_1, param_set_2):
        if pa.ndim == 1:
            continue
        pa_flat: th.Tensor = pa.flatten(start_dim=1)
        pb_flat: th.Tensor = pb.flatten(start_dim=1)
        cosines.append(cosine(pa_flat, pb_flat, dim=0).mean().item())
    mean_cos: float = mean(cosines)
    return mean_cos


"""PyTorchfunctional SGD Optimiser with rthogonalisation.

Modified from https://github.com/pytorch/pytorch/blob/5824a866b72c251ad47a9c16dc652e49cfd7e234/torch/optim/_functional.py
See the PyTorch License included.
"""

import torch
from torch import Tensor
from typing import List, Optional


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        orth: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]

        if orth and param.ndim > 1:
            d_p_flat = d_p.flatten(start_dim=1)

            try:
                u, _, v = torch.linalg.svd(d_p_flat, full_matrices=False)
            except RuntimeError:
                print("Failed to perform svd, adding some noise.")
                u, _, v = torch.linalg.svd(
                    d_p_flat + 1e-4 * d_p_flat.mean() * torch.randn_like(d_p_flat),
                    full_matrices=False)

            d_p = (u @ v).reshape_as(param)

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

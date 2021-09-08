from typing import Tuple
from typing import List

import logging
from math import sqrt
from statistics import mean

import torch
from torch.nn.functional import cosine_similarity as cosine

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


def _is_outlier(val: float,
                dist: Tuple[float, float],
                sigma_threshold: float = 2,
                alpha: float = 0.05) -> Tuple[bool, Tuple[float, float]]:
    """Use EMA to detect outlyers."""
    mean, var = dist
    # init dist, var should be 0 but not practical
    if mean == var == 0:
        return False, (val, 1)
    if abs(val - mean) > sigma_threshold * sqrt(var):
        return True, dist
    delta = val - mean
    dist = mean + alpha * delta, (1 - alpha) * (var + alpha * delta**2)
    return False, dist


@torch.no_grad()
def _orth_grads(optimiser):
    # Orthogonalise the gradients using SVD
    for group in optimiser.param_groups:
        orth = group['orth']
        for i, p in enumerate(group['params']):
            if orth and p.grad is not None and p.ndim > 1:
                G = p.grad.flatten(start_dim=1)
                G_norm = G.norm()
                try:
                    u, s, vt = torch.linalg.svd(G, full_matrices=False)
                    orth_G = u @ vt
                except RuntimeError:
                    logger.error('Failed to perform SVD, adding some noise.')
                    try:
                        u, s, v = torch.svd_lowrank(
                            G,
                            q=1,  # assume rank is at least 1
                            M=1e-4 * G.mean() * torch.randn_like(G))
                        orth_G = u @ v.T
                    except RuntimeError:
                        logger.error(('Failed to perform SVD with noise,'
                                      ' skipping gradient orthogonalisation'))
                        return

                # # TODO implelenent skip based off:
                # #       - grad norm
                # #       - mean cosine
                # #       - frobenius norm
                # # Have to store the data in the param group
                frob = (orth_G.div(orth_G.norm()) - G.div(G.norm())).norm().item()
                # cos = cosine(G, orth_G, dim=0).mean().item()
                skip, dist = _is_outlier(frob, group['dist'][i])
                logger.debug((
                    f'{"No " if not skip else "  "}'
                    'Skip '
                    f'|O-G|:{frob:.3f} '
                    f'|O-G|~({dist[0]:.3f}, {dist[1]:.3f})'))

                group['dist'][i] = dist
                if not skip:
                    p.grad = orth_G.reshape_as(p)

                # cos = cosine(G, orth_G, dim=1).mean()
                # print(1 / G.numel())
                # print(0, cosine(G, orth_G, dim=0).mean())
                # print(1, cosine(G, orth_G, dim=1).mean())
                # print(f'|G|: {G_norm:.3f}, |O|: {orth_G.norm():.3f}, s: {s.mean():.3f}')
                # p.grad = orth_G.reshape_as(p)


def orthogonalise(cls):
    og_init = cls.__init__
    og_step = cls.step

    def new_init(self, *args, orth=False, **kwargs):
        # Add orth hyperparam to defaults
        og_init(self, *args, **kwargs)
        self.defaults['orth'] = orth
        for grp in self.param_groups:
            grp.setdefault('orth', orth)
            grp.setdefault(
                'dist',
                [(0, 0) for i in range(len(grp['params']))])

    def new_step(self, *args, **kwargs):
        # Orthogonalise the grads before the original optim's step method
        _orth_grads(self)
        og_step(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.step = new_step
    return cls


def hook():
    from inspect import isclass
    for mod in dir(torch.optim):
        if mod.startswith('_'):
            continue
        _optim = getattr(torch.optim, mod)
        if (
            isclass(_optim)
            and issubclass(_optim, torch.optim.Optimizer)
            and _optim is not torch.optim.Optimizer
        ):
            setattr(torch.optim, mod, orthogonalise(_optim))


def hook_mmcv():
    hook()
    import mmcv
    # Re-register the optimisers from torch since they are now the orth versions
    mmcv.runner.optimizer.builder.register_torch_optimizers()

    # mmcv.runner.optimizer.builder.OPTIMIZERS.register_module(
    #     name='SGD',
    #     module=_orth_sgd,
    #     force=True)

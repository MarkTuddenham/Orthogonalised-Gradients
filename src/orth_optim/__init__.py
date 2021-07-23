import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


@torch.no_grad()
def _orth_grads(optimiser):
    # Orthogonalise the gradients using SVD
    for group in optimiser.param_groups:
        orth = group['orth']
        for p in group['params']:
            if orth and p.grad is not None and p.ndim > 1:
                d_p_flat = p.grad.flatten(start_dim=1)
                try:
                    u, _, vt = torch.linalg.svd(d_p_flat, full_matrices=False)
                    p.grad = (u @ vt).reshape_as(p)
                except RuntimeError:
                    logger.error('Failed to perform SVD, adding some noise.')
                    try:
                        u, _, v = torch.svd_lowrank(
                            d_p_flat,
                            q=1,  # assume rank is at least 1
                            M=1e-4 * d_p_flat.mean() * torch.randn_like(d_p_flat))
                        p.grad = (u @ v.T).reshape_as(p)
                    except RuntimeError:
                        logger.error(('Failed to perform SVD with noise,'
                                      ' skipping gradient orthogonalisation'))


def orthogonalise(cls):
    og_init = cls.__init__
    og_step = cls.step

    def new_init(self, *args, orth=False, **kwargs):
        # Add orth hyperparam to defaults
        og_init(self, *args, **kwargs)
        self.defaults['orth'] = orth
        for grp in self.param_groups:
            grp.setdefault('orth', orth)

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
        if (isclass(_optim)
           and issubclass(_optim, torch.optim.Optimizer)
           and _optim is not torch.optim.Optimizer):
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

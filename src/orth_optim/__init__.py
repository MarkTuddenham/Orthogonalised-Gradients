import torch


def _orth_grads(optimiser):
    # Orthogonalise the gradients before the original optim's step method
    for group in optimiser.param_groups:
        orth = group['orth']
        for p in group['params']:

            if orth and p.grad is not None and p.ndim > 1:
                d_p_flat = p.grad.flatten(start_dim=1)

                try:
                    u, _, vt = torch.linalg.svd(d_p_flat, full_matrices=False)
                    p.grad = (u @ vt).reshape_as(p)
                except RuntimeError:
                    print("Failed to perform SVD, adding some noise.")
                    try:
                        u, _, v = torch.svd_lowrank(
                            d_p_flat,
                            q=1,  # assume rank is at least 1
                            M=1e-4 * d_p_flat.mean() * torch.randn_like(d_p_flat))
                        p.grad = (u @ v.T).reshape_as(p)
                    except RuntimeError:
                        print('Failed to perform SVD with noise, using normal SGD')


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
        _orth_grads(self)
        og_step(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.step = new_step
    return cls


_orth_sgd = orthogonalise(torch.optim.SGD)


def hook():
    setattr(torch.optim, 'SGD', _orth_sgd)


def hook_mmcv():
    import mmcv
    mmcv.runner.optimizer.builder.OPTIMIZERS.register_module(
        name='SGD',
        module=_orth_sgd,
        force=True)

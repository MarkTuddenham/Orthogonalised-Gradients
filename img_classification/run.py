from typing import Dict
from typing import Any

import os
import logging
from dotenv import load_dotenv
from threading import Thread

from tqdm.auto import tqdm
from tqdm.auto import trange

import torch as th
from torch.nn import functional as F

from orth_optim import hook
from orth_optim import logger as orth_logger

from networks import BasicCNN_IR
from analysis import do_analysis
from analysis import logger as analysis_logger
from persist import save_tensor
from profiler import profile
from utils import get_device
from utils import get_weights
from utils import get_gradients
from utils import clone_gradients
from utils import run_data
from utils import models
from utils import logger as utils_logger
from utils import cosine_compare
from utils import get_save_opts_from_args
from utils import get_args

hook()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

th.set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=200,
    profile=None,
    sci_mode=None
)

load_dotenv('../')
load_dotenv()


def save_data(
        data: Dict[str, Any],
        save_opts,
        overwrite: bool = False,
        analysis: bool = False,
        full_analysis: bool = False):
    for k, v in data.items():
        save_tensor(v, 'results/' + k, save_opts, overwrite)
    if analysis:
        Thread(target=do_analysis, args=(save_opts, full_analysis)).start()


def get_optim(parameters, args):
    if args.adam == 0:
        optimiser = th.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nest,
            orth=args.orth,
            norm=args.norm)
    else:
        optimiser = th.optim.Adam(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.momentum, args.adam),
            orth=args.orth,
            norm=args.norm)
    return optimiser


def get_data_loaders(args, log_f):
    logger.debug('Getting data loaders')
    if args.dataset == 'cifar10':
        from data.cifar10 import set_data_path
        from data.cifar10 import get_train_gen
        from data.cifar10 import get_valid_gen
        from data.cifar10 import get_test_gen

    elif args.dataset == 'imagenet':
        from data.imagenet_hdf5 import set_data_path
        from data.imagenet_hdf5 import get_train_gen
        from data.imagenet_hdf5 import get_valid_gen
        from data.imagenet_hdf5 import get_test_gen
        # from data.imagenet_hdf5 import input_size
        from data.imagenet_hdf5 import logger as imagenet_hdf5_logger
        imagenet_hdf5_logger.addHandler(log_f)
    else:
        logger.error((f'Unknown dataset: {args.dataset},'
                      'please use either \'cifar10\' or \'imagenet\''))

    set_data_path(os.environ.get('DATA_PATH', './datasets'))
    train_loader = get_train_gen(args.batch_size)
    valid_loader = get_valid_gen(args.batch_size)
    test_loader = get_test_gen(args.batch_size)
    logger.debug('Got data loaders')
    return train_loader, valid_loader, test_loader


def train(model, device, args, log_f):
    save_opts = get_save_opts_from_args(args)
    logger.info(save_opts)

    # ==== Metric Containers ====
    data_collectors = {
        'model': None,
        'grad_norm': [],
        'cosine': [],
        'frobenius': [],
        'valid_losses': [],
        'valid_accuracies': [],
        'train_losses': [],
        'train_accuracies': [],
    }

    if args.layers is not None:
        logger.info('Saving layers %s', args.layers)
        for layer_name in args.layers:
            data_collectors[f'filter_path_{layer_name}'] = []
            data_collectors[f'filter_grad_path_{layer_name}'] = []

    if isinstance(model, BasicCNN_IR):
        data_collectors['IR'] = []
    # ==== / Metric Containers ====

    if args.save:
        save_data(data_collectors, save_opts)  # Create a new save to be overwritten later

    train_loader, valid_loader, test_loader = get_data_loaders(args, log_f)

    optimiser = get_optim(model.parameters(), args)

    lr_sched = None
    if args.schedule is not None:
        lr_sched = th.optim.lr_scheduler.MultiStepLR(
            optimiser,
            milestones=list(map(int, args.schedule)))

    train_loop(model,
               optimiser,
               train_loader,
               valid_loader,
               device,
               data_collectors,
               args,
               save_opts,
               lr_sched)

    test_loss, test_acc = run_data(model, device, test_loader, valid=False)
    save_tensor(th.tensor([test_loss, test_acc]), 'results/test_stats', save_opts)
    logger.info(f'Test Loss: {test_loss: .3f}, Acc: {test_acc: .3f}')
    if args.save:
        save_data(data_collectors, save_opts, overwrite=True, analysis=True, full_analysis=True)


def train_loop(model,
               optimiser,
               train_loader,
               valid_loader,
               device,
               data_collectors,
               args,
               save_opts,
               lr_sched=None):
    if args.avoid_tqdm:
        epoch_bar = range(1, args.epochs + 1)
    else:
        epoch_bar = trange(1, args.epochs + 1)
        epoch_bar.set_description('Epoch The Zeroth')
    logger.info('Epoch The Zeroth')

    for epoch in epoch_bar:
        train_loss, train_accuracy = do_epoch(
            args,
            model,
            optimiser,
            train_loader,
            device,
            data_collectors)

        if args.schedule is not None:
            lr_sched.step()

        # ==== Calculate metrics ====
        model.eval()
        train_loss /= len(train_loader.dataset)
        train_accuracy /= len(train_loader.dataset)
        valid_loss, valid_accuracy = run_data(model, device,
                                              valid_loader, valid=True)
        data_collectors['train_losses'].append(train_loss)
        data_collectors['train_accuracies'].append(train_accuracy)
        data_collectors['valid_losses'].append(valid_loss)
        data_collectors['valid_accuracies'].append(valid_accuracy)

        if args.save:
            data_collectors['model'] = model.state_dict()
            save_data(data_collectors, save_opts, overwrite=True, analysis=epoch % 10 == 0)
        # ==== / Calculate metrics ====

        epoch_status_str = (f"Epoch {epoch}: Valid Loss {valid_loss:.3f}, "
                            f"Accuracy {valid_accuracy:.2%}; "
                            f"Train Loss {train_loss:.3f}, "
                            f"Accuracy {train_accuracy:.2%}")
        logger.info(epoch_status_str)
        if not args.avoid_tqdm:
            epoch_bar.set_description(epoch_status_str)


def do_epoch(args, model, optimiser, train_loader, device, data_collectors):
    train_loss = 0
    train_accuracy = 0
    batch_count = len(train_loader)
    if args.avoid_tqdm:
        pbar = enumerate(train_loader)
    else:
        pbar = tqdm(
            enumerate(train_loader),
            leave=False,
            total=batch_count)

    model.train()
    # weights = get_weights(model)
    for batch_idx, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        z = model(x)

        L = F.cross_entropy(z, y)
        L.backward()

        if th.isnan(L).any():
            logger.warning('NaN loss, skipping batch')
            continue

        train_loss += L.item() * y.shape[0]
        pred = z.argmax(dim=1, keepdim=True)
        train_accuracy += pred.eq(y.view_as(pred)).sum().item()

        # Get IR if we are using BasicCNN_IR
        # Since we're saving a lot of data, only append every 10 batches
        if isinstance(model, BasicCNN_IR) and batch_idx % 10 == 0:
            data_collectors['IR'].append(model.IR)

        if args.layers is not None:
            for layer_name in args.layers:
                if isinstance(model, th.nn.DataParallel):
                    layer = eval("m." + layer_name, {'m': model.module})
                else:
                    layer = eval("m." + layer_name, {'m': model})

                data_collectors[f'filter_path_{layer_name}'].append(
                    layer.weight.detach().clone().cpu())
                data_collectors[f'filter_grad_path_{layer_name}'].append(
                    layer.weight.grad.detach().clone().cpu())

        og_grads = clone_gradients(model, concat=False, preserve_shape=True)
        # og_grads_vec = clone_gradients(model) but avoids making two clones
        og_grads_vec = th.cat([p.flatten() for p in og_grads])

        optimiser.step()

        orth_grads = get_gradients(model, concat=False, preserve_shape=True)
        orth_grads_vec = get_gradients(model)

        # Add grad norm data
        # avg cosine (grads before to grads after)
        # Frobenius norm of O-G -- need to keep original shape
        grad_norm = og_grads_vec.norm()
        cosine = cosine_compare(og_grads, orth_grads)
        frobenius = (orth_grads_vec - og_grads_vec).norm()
        data_collectors['grad_norm'].append(grad_norm)
        # TODO: think theres a problem here -> when orth is skipped, cosine is not 1
        data_collectors['cosine'].append(cosine)
        data_collectors['frobenius'].append(frobenius)
        # logger.debug(f'Frobenius: {frobenius: .3f}, |g|: {grad_norm: .3f}, cosine: {cosine: .3f}')

        status_str = (
            f"Batch {batch_idx}/{batch_count}: Loss {L: .3f}, "
            f"Running Epoch Acc: {train_accuracy / ((batch_idx + 1) * args.batch_size):.3%}, "
            f"|model|: {get_weights(model).norm():.3f}")  # TODO this should work as a ref
        if not args.avoid_tqdm:
            pbar.set_description(status_str)
        logger.debug(status_str)

    return train_loss, train_accuracy


@profile()
def main():
    log_dir = os.environ.get('LOG_DIR', './logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_f = logging.FileHandler(f'{os.path.dirname(log_dir)}/run.log', encoding='utf-8')
    log_f.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s: %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')
    log_f.setFormatter(formatter)
    logger.addHandler(log_f)
    utils_logger.addHandler(log_f)
    orth_logger.addHandler(log_f)
    analysis_logger.addHandler(log_f)

    args = get_args()
    device = get_device(args.no_cuda)

    model = models[args.model]().to(device)
    logger.info(args)
    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        logger.info(f'Using DataParallel with {len(args.gpu_ids)} GPUs: {" ".join(args.gpu_ids)}')
        model = th.nn.DataParallel(model, device_ids=list(map(int, args.gpu_ids)))

    # # Don't use with DataParallel either
    # if not args.model.startswith('densenet'):
    #     summary(model, input_size, device=device)

    train(model, device, args, log_f)


if __name__ == '__main__':
    main()

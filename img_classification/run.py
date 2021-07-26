import os
import argparse
import logging
from copy import copy
from dotenv import load_dotenv
from threading import Thread

from tqdm.auto import tqdm
from tqdm.auto import trange

import torch as th
from torch.nn import functional as F

# from torchsummary import summary

from orth_optim import hook

from analysis import do_analysis

from persist import save_tensor

from utils import get_device
from utils import get_weights
from utils import run_data
from utils import models
from utils import logger as utils_logger

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


def get_args():
    parser = argparse.ArgumentParser(description='Image Classification w/ SGD and Orhtogonalised SGD')
    parser.add_argument('--batch-size', '--bs', type=int, default=2**10, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', '--eps', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', '--mom', type=float, default=0.9, metavar='mom',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0,
                        metavar='WD', help='weight decay (default: 0)')
    parser.add_argument('--model', '-m', default='BasicCNN',
                        help='Name of the model to use (default: BasicCNN)')
    parser.add_argument('--orth', '-o', action='store_true', default=False,
                        help='Use orthogonalised SGD (defualt: False)')
    parser.add_argument('--dataset', default='cifar10',
                        help='Which data set to train on: cifar10/imagenet (default: cifar10)')

    parser.add_argument('--no-cuda', '-nc', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--gpu-ids', nargs='*',
                        help='List the gpu ids to train on (optional, e.g. --gpu-ids 0 1 2 3 4)')
    parser.add_argument('--save', '-s', action='store_true', default=False,
                        help='Save outputs? (default: False)')
    parser.add_argument('--avoid-tqdm', action='store_true', default=False,
                        help='Use tqdm progress bars or simple print (default: False)')
    parser.add_argument('--do-analysis', '-a', action='store_true', default=True,
                        help='Run the analysis while training (default: True)')
    args = parser.parse_args()
    return args


def train_loop(model, device, args, log_f):
    save_opts = copy(vars(args))
    del save_opts['save']
    del save_opts['no_cuda']
    del save_opts['avoid_tqdm']
    del save_opts['gpu_ids']
    del save_opts['do_analysis']
    logger.info(save_opts)

    # ==== Metric Containers ====
    valid_losses = []
    valid_accuracies = []
    train_losses = []
    train_accuracies = []
    data_collectors = {}
    # ==== / Metric Containers ====

    def save(overwrite=False, full_analysis=False, epoch=None):
        if args.save:
            save_tensor(model.state_dict(), 'results/model', save_opts, overwrite)
            save_tensor(valid_losses, 'results/valid_losses', save_opts, overwrite)
            save_tensor(valid_accuracies, 'results/valid_accuracies', save_opts, overwrite)
            save_tensor(train_losses, 'results/train_losses', save_opts, overwrite)
            save_tensor(train_accuracies, 'results/train_accuracies', save_opts, overwrite)
            if args.do_analysis and (epoch is None or epoch % 10 == 0):
                Thread(target=do_analysis, args=(save_opts, full_analysis)).start()
    save()  # Create a new save to be overwritten later

    logger.debug('Getting data loaders')
    if args.dataset == 'cifar10':
        from data.cifar10 import set_data_path
        from data.cifar10 import get_train_gen
        from data.cifar10 import get_valid_gen
        from data.cifar10 import get_test_gen
        from data.cifar10 import input_size

    elif args.dataset == 'imagenet':
        from data.imagenet_hdf5 import set_data_path
        from data.imagenet_hdf5 import get_train_gen
        from data.imagenet_hdf5 import get_valid_gen
        from data.imagenet_hdf5 import get_test_gen
        # from data.imagenet_hdf5 import input_size
        from data.imagenet_hdf5 import logger as imagenet_hdf5_logger
        imagenet_hdf5_logger.addHandler(log_f)
    else:
        logger.error(f'Unknown dataset: {args.dataset}, please use either \'cifar10\' or \'imagenet\'')

    set_data_path(os.environ.get('DATA_PATH', './datasets'))
    train_loader = get_train_gen(args.batch_size)
    valid_loader = get_valid_gen(args.batch_size)
    test_loader = get_test_gen(args.batch_size)
    logger.debug('Got data loaders')

    optimiser = th.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
        orth=args.orth)

    # lr_sched = th.optim.lr_scheduler.MultiStepLR(
    #     optimiser,
    #     milestones=[35, 75])

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

        # lr_sched.step()

        # ==== Calculate metrics ====
        model.eval()
        train_loss /= len(train_loader.dataset)
        train_accuracy /= len(train_loader.dataset)
        valid_loss, valid_accuracy = run_data(model, device,
                                              valid_loader, valid=True)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        save(overwrite=True, epoch=epoch)
        # ==== / Calculate metrics ====

        epoch_status_str = (f"Epoch {epoch}: Valid Loss {valid_loss:.3f}, "
                            f"Accuracy {valid_accuracy:.2%}; "
                            f"Train Loss {train_loss:.3f}, "
                            f"Accuracy {train_accuracy:.2%}")
        if not args.avoid_tqdm:
            epoch_bar.set_description(epoch_status_str)
        logger.info(epoch_status_str)

    run_data(model, device, test_loader, valid=False)
    save(overwrite=True, full_analysis=True)


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
    weights = get_weights(model)
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
        data_collectors['cosine'].append(cosine)
        data_collectors['frobenius'].append(frobenius)
        print(f'Frobenius: {frobenius: .3f}, |g|: {grad_norm: .3f}, cosine: {cosine: .3f}')

        status_str = (
            f"Batch {batch_idx}/{batch_count}: Loss {L: .3f}, "
            f"Running Epoch Acc: {train_accuracy / ((batch_idx + 1) * args.batch_size):.3%}, "
            f"|model|: {weights.norm():.3f}")
        if not args.avoid_tqdm:
            pbar.set_description(status_str)
        logger.debug(status_str)

    return train_loss, train_accuracy


def main():

    log_dir = os.environ.get('LOG_DIR', './logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_f = logging.FileHandler(f'{os.path.dirname(log_dir)}/run.log', encoding='utf-8')
    log_f.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    log_f.setFormatter(formatter)
    logger.addHandler(log_f)
    utils_logger.addHandler(log_f)
    # analysis_logger.addHandler(log_f)

    args = get_args()
    device = get_device(args.no_cuda)

    model = models[args.model]().to(device)
    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        logger.info(f'Using DataParallel with {len(args.gpu_ids)} GPUs: {" ".join(args.gpu_ids)}')
        model = th.nn.DataParallel(model, device_ids=map(str, args.gpu_ids))

    # # Don't use with DataParallel either
    # if not args.model.startswith('densenet'):
    #     summary(model, input_size, device=device)

    train_loop(model, device, args, log_f)


if __name__ == '__main__':
    main()

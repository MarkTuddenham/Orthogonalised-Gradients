from typing import Dict
from typing import Tuple

from copy import deepcopy
from os import makedirs
import os
import sys

import numpy as np

import logging

import torch as th
from torch.nn.functional import cosine_similarity as cosine

from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

from persist import load_last_tensor
from persist import load_tensor

from profiler import profile

# from utils import get_device
from utils import models
# from utils import run_data
from utils import get_args
from utils import get_save_opts_from_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

mpl_use('Agg')

th.set_printoptions(
    precision=7,
    threshold=None,
    edgeitems=None,
    linewidth=200,
    profile=None,
    sci_mode=None
)

fig_size = 6  # 4
rect_fig_size = (fig_size * 1.618, fig_size)
rect_fig_size_thin = (fig_size * 1.618, fig_size / 1.618)
# rect_fig_size = (fig_size * 2, fig_size)
# rect_fig_size = (fig_size * 2 * 1.618, fig_size)
pic_type = '.png'

makedirs('./plots/', exist_ok=True)

# (Orth, Norm) -> str
optim_type_text: Dict[Tuple[bool, str], str] = {
    (False, ''): '                   ',
    (True, ''): ' w/ Orth           ',
    (False, 'norm'): ' w/ Norm           ',
    (False, 'comp_norm'): ' w/ Component Norm ',
}


def cosine_matrix(xs):
    n = len(xs) if type(xs) is list else xs.shape[0]
    C = th.zeros((n, n), device=xs[0].device)
    for i in range(n):
        for j in range(n):
            C[i][j] = cosine(xs[i].flatten(), xs[j].flatten(), dim=0)
    return C


def cosine_matrix_flat(xs):
    n = len(xs) if type(xs) is list else xs.shape[0]
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            cosines.append(cosine(xs[i].flatten(), xs[j].flatten(), dim=0))
    return th.tensor(cosines)


def calc_test_avg(opts, max_len=5):
    test_stats = load_tensor('results/test_stats', opts)
    if test_stats is None:
        return

    test_mean = test_stats[:, :max_len].mean(dim=0)
    test_std = test_stats[:, :max_len].std(dim=0)
    num_runs = test_stats[:, 0].numel()
    n = min(num_runs, max_len)
    test_stderr = test_std.div(np.sqrt(n))

    text = optim_type_text[opts['orth'], opts['norm']]
    logger.info((f'{opts["model"]}{text}- {n}/{num_runs} runs - Test acc: '
                 f'{test_mean[1]*100:.2f}+-{test_stderr[1]*100:.2f}, '
                 f'Loss: {test_mean[0]:.4f}+-{test_stderr[0]:.4f}'))


def setup_loss_acc_plot():
    losses_fig = plt.figure(figsize=rect_fig_size)
    losses_ax = losses_fig.add_subplot(111)
    acc_fig = plt.figure(figsize=rect_fig_size)
    acc_ax = acc_fig.add_subplot(111)
    sgd_cmap = plt.get_cmap('jet_r')
    adam_cmap = plt.get_cmap('PiYG')

    losses_ax.spines['right'].set_color(None)
    losses_ax.spines['top'].set_color(None)
    losses_ax.set_xlabel('Epoch')
    losses_ax.set_ylabel('Validation Loss')
    losses_ax.set_ylim((0, 3))

    acc_ax.spines['right'].set_color(None)
    acc_ax.spines['top'].set_color(None)
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Validation Accuracy (%)')

    return losses_fig, losses_ax, acc_fig, acc_ax, sgd_cmap, adam_cmap


def plot_loss_acc_for_model(i, model_c, opts, plot_items, full):
    valid_losses = load_last_tensor('results/valid_losses', opts)
    if not valid_losses:  # quick escape if there's no data
        return
    valid_accuracies = load_last_tensor('results/valid_accuracies', opts)

    _, losses_ax, _, acc_ax, sgd_cmap, adam_cmap = plot_items

    label_prefix = f'{opts["model"]} with '
    line_type = '--'

    if opts['orth']:
        label_prefix += 'Orth '
        line_type = '-'
    elif opts['norm'] == 'norm':
        label_prefix += 'Norm '
        line_type = ':'
    elif opts['norm'] == 'comp_norm':
        label_prefix += 'Comp Norm '
        line_type = ':'

    if opts['nest']:
        label_prefix += 'Nesterov '

    if opts['adam']:
        label = label_prefix + 'Adam'
        colour = adam_cmap(float(i) / len(models.keys()))
    else:
        label = label_prefix + 'SGDM'
        colour = sgd_cmap(float(i) / len(models.keys()))

    losses_ax.plot(valid_losses, line_type, c=colour, label=label)
    acc_ax.plot(valid_accuracies, line_type, c=colour, label=label)


def finalise_loss_acc_plot(plot_items):
    losses_fig, losses_ax, acc_fig, acc_ax, _, _ = plot_items
    losses_ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.26, 1))
    # losses_ax.legend(loc='lower right')
    losses_fig.savefig('plots/losses' + pic_type, bbox_inches='tight')
    plt.close(losses_fig)
    acc_ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.26, 1))
    # acc_ax.legend(loc='lower right')
    acc_fig.savefig('plots/accuracies' + pic_type, bbox_inches='tight')
    plt.close(acc_fig)


def plot_per_batch(opts):
    fig = plt.figure(figsize=rect_fig_size)
    ax = fig.add_subplot(111)

    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    # ax.set_ylim((0, 3))

    for i in range(2, 11):
        bs = 2**i
        opts.update({'batch_size': bs, 'orth': False})
        val_l = load_last_tensor('results/valid_losses', opts)
        if not val_l:
            continue
        opts.update({'orth': True})
        val_l_o = load_last_tensor('results/valid_losses', opts)
        if not val_l_o:
            continue
        val_l, val_l_o = th.tensor(val_l), th.tensor(val_l_o)
        last_ep = min(val_l.numel(), val_l_o.numel())
        diff = val_l_o[:last_ep] - val_l[:last_ep]

        ax.plot(diff, label=bs)

    ax.legend(loc='lower right', frameon=False)
    fig.savefig('plots/acc_per_batch' + pic_type, bbox_inches='tight')
    plt.close(fig)


# only does the resnet ones
layer_options = [
    'conv1',

    'layer1[0].conv1',
    'layer1[0].conv2',
    'layer1[1].conv1',
    'layer1[1].conv2',
    'layer1[2].conv1',
    'layer1[2].conv2',

    'layer2[0].conv1',
    'layer2[0].conv2',
    'layer2[1].conv1',
    'layer2[1].conv2',
    'layer2[2].conv1',
    'layer2[2].conv2',

    'layer3[0].conv1',
    'layer3[0].conv2',
    'layer3[1].conv1',
    'layer3[1].conv2',
    'layer3[2].conv1',
    'layer3[2].conv2',
]


def dead_relus(opts):
    for layer in layer_options:
        filter_grad_path = load_last_tensor(f'results/filter_grad_path_{layer}', opts)
        if filter_grad_path is None:  # no results for this network
            continue

        x_size = len(filter_grad_path)
        s = filter_grad_path[0].numel()

        eps = 1e-4
        num_dead = []
        for i in range(x_size):
            zeroed_grad_path = th.where(
                filter_grad_path[i].abs() > eps,
                filter_grad_path[i],
                th.tensor(0.))
            num_dead.append(s - zeroed_grad_path.count_nonzero())
        num_dead = th.tensor(num_dead).float()

        text = optim_type_text[opts['orth'], opts['norm']]
        logger.info((f'{opts["model"]} {layer} {text}- '
                     f'Dead Neurons: {num_dead.mean(): .3e} '
                     f'+-{num_dead.var(): .3e} out of {s: .3e}'))

        N = 50
        running_range = range(N // 2, x_size - N // 2 + 1)
        running_count = np.convolve(num_dead, np.ones(N) / N, mode='valid')

        fig = plt.figure(figsize=rect_fig_size)
        ax = fig.add_subplot(111)

        ax.scatter(range(x_size),
                   num_dead,
                   color='blue',
                   marker='.',
                   alpha=0.25,
                   label='number of dead neurons')
        ax.plot(running_range,
                running_count,
                color='orange',
                ls='-',
                label=f'running count (N={N})')

        ax.spines['right'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.set_xlabel('Batch #')
        ax.set_ylabel(f'count ($eps={eps}$)')

        ax_leg = ax.legend(frameon=False)
        for lh in ax_leg.legendHandles:
            lh.set_alpha(1)

        name = f'plots/dead_relu_{opts["model"]}_{layer}_{opts["orth"]}' f'{pic_type}'
        logger.debug(f'Saved plot {name}')
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

        del num_dead
        del filter_grad_path


def plot_filter_cosimilarity(opts):
    for layer in layer_options:
        skip = 10

        fig = plt.figure(figsize=rect_fig_size_thin)
        ax = fig.add_subplot(111)
        ax.spines['right'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.set_xlabel(f'Batch # /{skip}s')
        ax.set_ylabel('abs(cosine)')
        # ax.axhline(0, ls='-', lw=0.5, alpha=0.5, color='k')

        colours = ['blue', 'orange']

        should_save = False

        for orth in [False, True]:
            # filter_x_path = load_last_tensor(f'results/filter_grad_path_{layer}',
            #         {**opts, 'orth': orth})
            filter_x_path = load_last_tensor(f'results/filter_path_{layer}', {**opts, 'orth': orth})
            if filter_x_path is None:  # no results for this network
                continue

            should_save = True

            s = filter_x_path[0].numel()
            threshold = 4 / np.sqrt(s)

            mean_cosines = []
            var_cosines = []
            n = len(filter_x_path)
            for j in range(0, n, skip):
                cosines = cosine_matrix_flat(filter_x_path[j].cuda())
                mean_cosines.append(cosines.abs().mean().cpu())
                var_cosines.append(cosines.abs().var().cpu())

            x_size = len(mean_cosines)
            ax.set_xlim((0, x_size))

            # N = 10
            # running_range = range(N // 2, x_size - N // 2 + 1)
            # running_mean = np.convolve(mean_cosines, np.ones(N) / N, mode='valid')
            # running_var = np.convolve(var_cosines, np.ones(N) / N, mode='valid')

            text = optim_type_text[opts['orth'], opts['norm']]
            ax.plot(range(x_size), mean_cosines, color=colours[orth], label='mean' + text)
            # ax.plot(running_range, running_mean, color='blue', ls='-', label='running mean')
            # ax.scatter(range(x_size), var_cosines, color='orange',
            #                    marker='+', alpha=0.5, label='variance')
            # ax.plot(running_range, running_var, color='orange', ls='--', label='running var')

            del filter_x_path
            del mean_cosines
            del var_cosines
            # del running_mean
            # del running_var

        # cosines_ax.set_ylim((-0.1, 1))
        # cosines_ax.set_ylim((-0.01, 0.02))

        if should_save:
            ax.axhline(threshold, ls='--', color='k', label='cosine threshold')

            cosines_ax_leg = ax.legend(frameon=False)
            for lh in cosines_ax_leg.legendHandles:
                lh.set_alpha(1)

            name = f'plots/cosines_{opts["model"]}_{layer}{pic_type}'
            logger.debug(f'Saved plot {name}')
            fig.savefig(name, bbox_inches='tight')
        plt.close(fig)


def ir_cosimilarity(opts):
    fig = plt.figure(figsize=rect_fig_size)
    ax = fig.add_subplot(111)
    ax.set_ylim((0, 0.62))
    # cosines_ax.set_ylim((-0.1, 1))
    # cosines_ax.set_ylim((-0.01, 0.02))

    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.set_xlabel('Batch # /10s')
    ax.set_ylabel('abs(cosine)')

    # ax.axhline(0, ls='-', lw=0.5, alpha=0.5, color='k')

    colours = ['blue', 'orange']

    should_save = False

    for orth in [False, True]:
        irs = load_last_tensor('results/IR', {**opts, 'orth': orth})
        if irs is None:  # no results for this network
            continue

        should_save = True
        num_irs = len(irs[0])

        var_cosines = []
        mean_cosines = []
        minibatch_count = len(irs)
        for mb in range(minibatch_count):
            for ir in range(num_irs):
                minibatch_ir = irs[mb][ir]
                for sample in range(len(minibatch_ir)):
                    cosines = cosine_matrix_flat(minibatch_ir[sample].cuda())
                    mean_cosines.append(cosines.abs().mean().cpu())
                    var_cosines.append(cosines.abs().var().cpu())

        logger.info('ir_cosimilarity opts: %s', list(map(opts.get, ['model', 'orth'])))
        logger.info(f'cosines : {np.mean(mean_cosines) :.3f}+-{np.var(var_cosines):.3f}')
        logger.info(
            f'all cosine : mean: {np.mean(mean_cosines) :.3f} var: {np.mean(var_cosines):.3f}')
        logger.info(f'last cosine : {mean_cosines[-1]:.3f} var: {var_cosines[-1]:.3f}')

        x_size = len(mean_cosines)

        # N = 10
        # running_range = range(N // 2, x_size - N // 2 + 1)
        # running_mean = np.convolve(mean_cosines, np.ones(N) / N, mode='valid')
        # running_var = np.convolve(var_cosines, np.ones(N) / N, mode='valid')
        text = optim_type_text[opts['orth'], opts['norm']]
        ax.scatter(range(x_size), mean_cosines, color=colours[orth],
                   marker='.', alpha=0.4, linewidths=0, label='mean' + text)
        # cosines_ax.plot(running_range, running_mean, color='blue', ls='-', label='running mean')
        # ax.scatter(range(x_size), var_cosines, color='orange',
        #                    marker='+', alpha=0.25, label='variance')
        # cosines_ax.plot(running_range, running_var, color='orange', ls='--', label='running var')

        del irs
        del mean_cosines
        del var_cosines

    if should_save:
        ax.set_xlim((0, x_size))

        cosines_ax_leg = ax.legend(frameon=False)
        for lh in cosines_ax_leg.legendHandles:
            lh.set_alpha(1)

        name = f'plots/IR_cosines_{opts["model"]}' f'{pic_type}'
        logger.debug(f'Saved plot {name}')
        fig.savefig(name, bbox_inches='tight')
    plt.close(fig)


@profile()
def do_analysis(opts, full=False):
    logger.info('====== Analysis ======')
    opts = deepcopy(opts)
    logger.info('Starting opts: %s', opts)

    if full:
        # plot_per_batch(opts)

        plot_filter_cosimilarity(opts)

        dead_relus(opts)
        dead_relus({**opts, 'orth': not opts['orth']})

        ir_cosimilarity(opts)

    plot_items = setup_loss_acc_plot()
    for i, (model_name, model_c) in enumerate(models.items()):
        logger.info(f'Analysing results for {model_name}')

        opts.update({'model': model_name})

        for adam in [0, 0.99]:
            opts.update({'adam': adam})
            for orth, norm in [
                    (False, ''),
                    (False, 'norm'),
                    (False, 'comp_norm'),
                    (True, '')]:
                opts.update({'orth': orth, 'norm': norm})
                if full:
                    calc_test_avg(opts)
                plot_loss_acc_for_model(i, model_c, opts, plot_items, full)

    finalise_loss_acc_plot(plot_items)


if __name__ == '__main__':
    log_dir = os.environ.get('LOG_DIR', './logs/')
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s: %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')
    log_stdout = logging.StreamHandler(sys.stdout)
    log_stdout.setLevel(logging.INFO)
    log_stdout.setFormatter(formatter)

    log_f = logging.FileHandler(f'{os.path.dirname(log_dir)}/results.log', encoding='utf-8')
    log_f.setLevel(logging.DEBUG)
    log_f.setFormatter(formatter)
    logger.addHandler(log_stdout)
    logger.addHandler(log_f)

    opts = get_save_opts_from_args(get_args())
    do_analysis(opts, True)

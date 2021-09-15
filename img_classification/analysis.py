from copy import deepcopy
from os import makedirs
import os
import sys

import math as maths  # :)

import logging

import torch as th
# from torch.nn.functional import cosine_similarity as cosine

from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

from persist import load_last_tensor
from persist import load_tensor

# from utils import get_device
from utils import models
# from utils import run_data

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
# rect_fig_size = (fig_size * 2, fig_size)
# rect_fig_size = (fig_size * 2 * 1.618, fig_size)
pic_type = '.png'

makedirs('./plots/', exist_ok=True)


def calc_test_avg(opts, max_len=5):
    test_stats = load_tensor('results/test_stats', opts)
    if test_stats is None:
        return

    test_mean = test_stats[:, :max_len].mean(dim=0)
    test_std = test_stats[:, :max_len].std(dim=0)
    num_runs = test_stats[:, 0].numel()
    n = min(num_runs, max_len)
    test_stderr = test_std.div(maths.sqrt(n))

    orth_text = ' w/ Orth ' if opts['orth'] else '         '
    logger.info(f'{opts["model"]}{orth_text}- {n}/{num_runs} runs - Test acc: {test_mean[1]*100:.2f}+-{test_stderr[1]*100:.2f}, Loss: {test_mean[0]:.4f}+-{test_stderr[0]:.4f}')


def setup_loss_acc_plot():
    losses_fig = plt.figure(figsize=rect_fig_size)
    losses_ax = losses_fig.add_subplot(111)
    acc_fig = plt.figure(figsize=rect_fig_size)
    acc_ax = acc_fig.add_subplot(111)
    cmap = plt.get_cmap('jet_r')

    losses_ax.spines['right'].set_color(None)
    losses_ax.spines['top'].set_color(None)
    losses_ax.set_xlabel('Epoch')
    losses_ax.set_ylabel('Validation Loss')
    losses_ax.set_ylim((0, 3))

    acc_ax.spines['right'].set_color(None)
    acc_ax.spines['top'].set_color(None)
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Validation Accuracy (%)')

    return losses_fig, losses_ax, acc_fig, acc_ax, cmap


def plot_loss_acc_for_model(i, model_c, opts, plot_items, full):
    valid_losses = load_last_tensor('results/valid_losses', opts)
    if not valid_losses:  # quick escape if there's no data
        return
    valid_accuracies = load_last_tensor('results/valid_accuracies', opts)

    _, losses_ax, _, acc_ax, cmap = plot_items

    colour = cmap(float(i) / len(models.keys()))

    label_prefix = f'{opts["model"]} with '
    if opts['orth']:
        label_prefix += 'Orth '
    if opts['nest']:
        label_prefix += 'Nesterov '
    label = label_prefix + 'SGD'

    # TODO plot from 1 on the x axis
    line_type = '.-' if opts['nest'] else '--'
    if opts['orth']:
        line_type = ':' if opts['nest'] else '-'

    losses_ax.plot(valid_losses, line_type, c=colour, label=label)
    acc_ax.plot(valid_accuracies, line_type, c=colour, label=label)


def finalise_loss_acc_plot(plot_items):
    losses_fig, losses_ax, acc_fig, acc_ax, _ = plot_items
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

    ax.legend(loc='lower right')
    fig.savefig('plots/acc_per_batch' + pic_type, bbox_inches='tight')
    plt.close(fig)


def do_analysis(opts, full=False):
    logger.info('====== Analysis ======')
    opts = deepcopy(opts)
    logger.info('Starting opts: %s', opts)

    # plot_per_batch(opts)

    plot_items = setup_loss_acc_plot()
    for i, (model_name, model_c) in enumerate(models.items()):
        logger.info(f'Analysing results for {model_name}')
        opts.update({'model': model_name, 'orth': False})
        if full:
            calc_test_avg(opts)
        plot_loss_acc_for_model(i, model_c, opts, plot_items, full)
        opts.update({'orth': True})
        if full:
            calc_test_avg(opts)
        plot_loss_acc_for_model(i, model_c, opts, plot_items, full)
    finalise_loss_acc_plot(plot_items)


if __name__ == '__main__':
    log_dir = os.environ.get('LOG_DIR', './logs/')
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s: %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')
    log_std = logging.StreamHandler(sys.stdout)
    log_std.setLevel(logging.INFO)
    log_std.setFormatter(formatter)

    log_f = logging.FileHandler(f'{os.path.dirname(log_dir)}/results.log', encoding='utf-8')
    log_f.setLevel(logging.DEBUG)
    log_f.setFormatter(formatter)
    logger.addHandler(log_std)
    logger.addHandler(log_f)

    opts = {
        'batch_size': 1024,
        'epochs': 100,
        'learning_rate': 1e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'model': 'resnet18',
        'orth': False,
        'nest': False,
        'dataset': 'cifar10',
        'layer': 'conv1'
    }
    do_analysis(opts, True)

from os import mkdir
from os.path import isdir
from copy import deepcopy

import numpy as np

import torch as th
from torch.nn.functional import cosine_similarity as cosine

from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

from persist import load_last_tensor
from utils import models

mpl_use('Agg')

th.set_printoptions(
    precision=7,
    threshold=None,
    edgeitems=None,
    linewidth=200,
    profile=None,
    sci_mode=None
)

fig_size = 5
# rect_fig_size = (fig_size * 1.618, fig_size)
# rect_fig_size = (fig_size*2, fig_size)
rect_fig_size = (fig_size * 2 * 1.618, fig_size)
pic_type = '.png'

test_loader = None

if not isdir("plots/"):
    mkdir("plots/")


def cosine_matrix(xs):
    n = len(xs) if type(xs) is list else xs.shape[0]
    C = th.zeros((n, n), device=xs[0].device)
    for i in range(n):
        for j in range(n):
            C[i][j] = cosine(xs[i].flatten(), xs[j].flatten(), dim=0)
    return C


def plot_filter_cosimilarity_hist(opts, on_grads=True, alpha=0.25):
    path = 'filter_grad_path' if on_grads else 'filter_path'
    filter_x_path = load_last_tensor(f'results/{path}', opts)
    if filter_x_path is None:  # no results for this network
        return

    n = len(filter_x_path)
    cosines = []
    intervals = [0, (n - 1) // 2, n - 1]
    for i in intervals:
        C = cosine_matrix(filter_x_path[i].cuda())
        s = C.shape[0]
        ind = th.triu_indices(s, s, 1)
        cosines.append(C[ind[0], ind[1]].cpu())

    cosines_fig = plt.figure(figsize=rect_fig_size)
    cosines_ax = cosines_fig.add_subplot(111)
    cosines_ax.set_xlim((-1, 1))

    colours = ['red', 'blue', 'green']
    bins = np.arange(-1, 1, 0.05)
    for c, i, colour in zip(cosines, intervals, colours):
        l = c.shape[0]
        cosines_ax.hist(c.tolist()[:l // 2], alpha=0.2, color=colour, bins=bins, label=f'{i+1}')

    # cosines_ax.set_ylim((-1, 1))

    cosines_ax.spines['right'].set_color(None)
    cosines_ax.spines['top'].set_color(None)
    cosines_ax.set_xlabel('cosine')
    cosines_ax.set_ylabel('count')

    cosines_ax_leg = cosines_ax.legend()
    for lh in cosines_ax_leg.legendHandles:
        lh.set_alpha(1)

    name = 'cosines_hist_grad' if on_grads else 'cosines_hist'
    if opts['orth']:
        name += '_orth'
    cosines_fig.savefig((f'plots/{name}_{opts["model"]}_{opts["layer"]}'
                         f'{pic_type}'), bbox_inches='tight')
    plt.close(cosines_fig)


def do_analysis(opts):
    opts = deepcopy(opts)
    for i, (model_name, model_c) in enumerate(models.items()):
        opts.update({'model': model_name, 'orth': False})
        plot_filter_cosimilarity_hist(opts)
        plot_filter_cosimilarity_hist(opts, on_grads=False)
        opts.update({'orth': True})
        plot_filter_cosimilarity_hist(opts)
        plot_filter_cosimilarity_hist(opts, on_grads=False)


if __name__ == '__main__':
    do_analysis({'batch_size': 1024,
                 'epochs': 100,
                 'learning_rate': 1e-2,
                 'momentum': 0.9,
                 'weight_decay': 5e-4,
                 'model': 'BasicCNN',
                 'orth': True,
                 'orth_extended': False,
                 'dataset': 'cifar10',
                 'layer': 'conv2'},
                )

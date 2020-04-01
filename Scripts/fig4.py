import pylab as pl
import numpy as np
from LNLmodel import det_gaussian, gaussian_kernel
from tools import cm2inch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# PLOT SETTINGS
# # # # # # # # # #

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=8)

markersize = 2
capsize = 2
lw = 0.75

c_gaussians = ('#984ea3', '#377eb8', '#4daf4a')
c_test = '#e41a1c'
c_zeroline = '#e41a1c'
c_data = 'black'

figsize_data = cm2inch(9.5, 5.5)
figsize_modelsteps = cm2inch(7.5, 10)  # 1x5 Grid
figsize_gaussians = cm2inch(8.5, 5)
figsize_modelres = cm2inch(9.5, 5.5)

t_modelsteps = 140000  # number of timesteps to plot in modelsteps plots
t_gaussians = 140000  # number of timesteps to plot in gaussian plot

example_Tafter = 19000  # T_after for example test spike in modelsteps plot

t_modelres = 14  # seconds to plot in model result plot


# DATA (estimated from Neubrandt et al. 2018)
# # # # # # # # # #

data_x = np.array([0.1, 0.5, 1.0, 2.3, 3.5, 4.8, 6.0, 7.3, 8.50, 9.50, 12.5])
data_y = np.array([2.1, 2.7, 3.4, 3.6, 3.1, 3.3, 2.8, 2.9, 2.25, 1.65, 1.20])
data_yerr = np.array([0.4, 0.5, 0.4, 0.4, 0.2, 0.3, 0.2, 0.6, 0.25, 0.3, 0.15])
data_xerr = np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.2, 0.5])


# MODEL PARAMETERS
# # # # # # # # # #

dt = 0.1  # ms per bin
T = 100e3  # in ms

t = np.arange(0, T, dt)
L = len(t)
spktr = np.zeros(t.shape)

b = -1.5

acommon = 0.25
mus = [1.5e3, 2.5e3, 6.0e3]
sigmas = [.5e3, 1.2e3, 3.0e3]
amps = np.array([350, 1800, 6500])

# SCRIPT
# # # # # # # # # #

ktot = gaussian_kernel(acommon, amps, mus, sigmas, T, dt)

k1 = gaussian_kernel(acommon, np.array([amps[0]]), np.array([mus[0]]), np.array([sigmas[0]]), T, dt)
k2 = gaussian_kernel(acommon, np.array([amps[1]]), np.array([mus[1]]), np.array([sigmas[1]]), T, dt)
k3 = gaussian_kernel(acommon, np.array([amps[2]]), np.array([mus[2]]), np.array([sigmas[2]]), T, dt)

lateSTF = det_gaussian(acommon, amps, mus, sigmas, b, T, dt)  # synapse object

# SPIKE TRAINS
spktr[[4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]] = 1  # BURST ONLY TRAIN
spktr_example = spktr.copy()  # Burst + One Test spike
spktr_example[[12000 + example_Tafter]] = 1

Trange = np.arange(1500, 125000, 5000)
ratio = np.zeros(Trange.shape)
for testspike in Trange:
    train = spktr.copy()
    train[[12000 + testspike]] = 1
    res = lateSTF.run(train)['nl_readout']
    index = np.where(Trange == testspike)[0]
    uctl = res[0]  # CONTROL SPIKE = BASELINE EFFICACY
    ulate = res[12001 + testspike]
    ratio[index] = ulate / uctl

res_burstonly = lateSTF.run(spktr)
res_example = lateSTF.run(spktr_example)


# PLOTTING FUNCTIONS
# # # # # # # # # #

def plot_data():
    fig, ax = plt.subplots(figsize=figsize_data)
    ax.errorbar(data_x, data_y, yerr=data_yerr, xerr=data_xerr,
                capsize=capsize, marker='s', lw=lw, markersize=markersize, color=c_data)
    ax.axhline(y=1, c='black', ls='dashed', lw=lw)

    ax.set_xlim(left=0, right=t_modelres)
    ax.set_ylim(bottom=0, top=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel('time after burst (s)')
    ax.set_ylabel('test/control EPSC amplitude')
    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    plt.tight_layout()

    return fig


def plot_modelres():
    fig, ax = plt.subplots(figsize=figsize_modelres)
    x_ax = Trange / 1000 * 0.1
    ax.plot(x_ax, ratio, c=c_test, lw=lw)
    ax.axhline(y=1, c='black', ls='dashed', lw=lw)

    ax.set_xlim(left=0, right=t_modelres)
    ax.set_ylim(bottom=0, top=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel('time after burst (s)')
    ax.set_ylabel('test/control PSP amplitude')

    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    plt.tight_layout()

    return fig


def plot_gaussians():
    fig, ax = plt.subplots(figsize=figsize_gaussians)
    x_ax = np.arange(t_gaussians) * 0.0001

    ax.plot(x_ax, ktot[:t_gaussians], c='black', lw=lw, ls='dashed')

    # ax.plot(k1[:t_gaussians], c=c_gaussians[0], lw=lw, zorder=3)
    ax.fill_between(x_ax, 0, k1[:t_gaussians], color=c_gaussians[0], alpha=0.6, zorder=3)

    # ax.plot(k2[:t_gaussians], c=c_gaussians[1], lw=lw, zorder=2)
    ax.fill_between(x_ax, 0, k2[:t_gaussians], color=c_gaussians[1], alpha=0.6, zorder=2)

    # ax.plot(k3[:t_gaussians], c=c_gaussians[2], lw=lw, zorder=1)
    ax.fill_between(x_ax, 0, k3[:t_gaussians], color=c_gaussians[2], alpha=0.6, zorder=1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, bottom=True)
    ax.set_xlabel('time after burst (s)')
    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    ax.yaxis.set_ticks([])
    plt.tight_layout()

    return fig


def plot_modelsteps():
    fig, axes = plt.subplots(figsize=figsize_modelsteps, ncols=1, nrows=5,
                             sharex=True, sharey=False)
    plt.subplots_adjust(hspace=1)
    for i in axes:
        i.axis('off')
    axes[0].plot(lateSTF.k[:t_modelsteps], c='black', lw=lw)
    axes[0].set_title('Kernel', loc='left')

    axes[1].plot(spktr_example[:t_modelsteps], c='black', lw=0.5)
    axes[1].set_title('Spiketrain', loc='left')

    axes[2].plot(res_burstonly['filtered_s'][:t_modelsteps], c='black', lw=lw)
    axes[2].axhline(y=0, c=c_zeroline, ls='dashed')
    axes[2].set_title('Filtered spiketrain', loc='left')

    axes[3].plot(res_burstonly['nl_readout'][:t_modelsteps], c='black', lw=lw)
    axes[3].set_title('Nonlinear readout', loc='left')

    axes[4].plot(res_example['efficacy'][:t_modelsteps], c='black', lw=0.5)
    axes[4].set_title('Synaptic efficacy', loc='left')

    return fig

def plot():

    return (plot_data(), plot_gaussians(), plot_modelres(), plot_modelsteps())


if __name__ == '__main__':
    import os, inspect

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)

    figure_modelsteps = plot_modelsteps()
    figure_modelsteps.savefig(parent_dir + '/Figures/Fig4_modelsteps.pdf')
    plt.close('all')

    figure_gaussians = plot_gaussians()
    figure_gaussians.savefig(parent_dir + '/Figures/Fig4_gaussians.pdf')
    plt.close('all')

    figure_modelres = plot_modelres()
    figure_modelres.savefig(parent_dir + '/Figures/Fig4_modelres.pdf')
    plt.close('all')

    figure_data = plot_data()
    figure_data.savefig(parent_dir + '/Figures/Fig4_data.pdf')
    plt.close('all')

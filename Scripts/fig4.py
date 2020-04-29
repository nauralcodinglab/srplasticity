import pylab as pl
import numpy as np
from LNLmodel import det_gaussian, gaussian_kernel
from tools import add_figure_letters, get_stimvec, get_ISIvec

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. PLOT SETTINGS
# # # # # # # # # #

plt.style.use('science')
matplotlib.rc('xtick', top=False)
matplotlib.rc('ytick', right=False)
matplotlib.rc('ytick.minor', visible=False)
matplotlib.rc('xtick.minor', visible=False)
plt.rc('font', size=8)

markersize = 2
capsize = 2
lw = 1
figsize = (5.25102, 5.25102 * (2 / 3))  # From LaTeX readout of textwidth

# COLORS
c_gaussians = ('#525252', '#969696', '#cccccc')  # greyscale
c_model = '#cc3311'
c_data = 'black'
c_zeroline = '#969696'
c_baseline = 'black'
c_burst = '#0077bb'

t_modelsteps = 140000  # number of timesteps to plot in modelsteps plots
t_gaussians = 140000  # number of timesteps to plot in gaussian plot
example_Tafter = 19000  # T_after for example test spike in modelsteps plot
t_modelres = 14  # seconds to plot in model result plot


# 2. DATA (estimated from Neubrandt et al. 2018) Figure B
# # # # # # # # # #

data_x = np.array([0.1, 0.5, 1.0, 2.3, 3.5, 4.8, 6.0, 7.3, 8.50, 9.50, 12.5])
data_y = np.array([2.1, 2.7, 3.4, 3.6, 3.1, 3.3, 2.8, 2.9, 2.25, 1.65, 1.20])
data_yerr = np.array([0.4, 0.5, 0.4, 0.4, 0.2, 0.3, 0.2, 0.6, 0.25, 0.3, 0.15])
data_xerr = np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.2, 0.5])

# 3. MODEL PARAMETERS
# # # # # # # # # #

dt = 0.1  # ms per bin
T = 100e3  # in ms
t = np.arange(0, T, dt)
L = len(t)
spktr = np.zeros(L)

b = -1.2
acommon = 0.2
mus = [1e3, 2.5e3, 6.0e3]
sigmas = [.6e3, 1.3e3, 2.8e3]
amps = np.array([450, 2200, 5500])

# INITIALIZE MODEL AND KERNELS
lateSTF = det_gaussian(acommon, amps, mus, sigmas, b, T, dt)  # synapse object
ktot = gaussian_kernel(acommon, amps, mus, sigmas, T, dt)
k1 = gaussian_kernel(acommon, np.array([amps[0]]), np.array([mus[0]]), np.array([sigmas[0]]), T, dt)
k2 = gaussian_kernel(acommon, np.array([amps[1]]), np.array([mus[1]]), np.array([sigmas[1]]), T, dt)
k3 = gaussian_kernel(acommon, np.array([amps[2]]), np.array([mus[2]]), np.array([sigmas[2]]), T, dt)


# 4. SCRIPT
# # # # # # # # # #


# Example spike train
spktr_burstonly = spktr.copy()
spktr_burstonly[[4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]] = 1  # BURST ONLY TRAIN
spktr_example = spktr_burstonly.copy()
spktr_example[[12000 + example_Tafter]] = 1 # Burst + One Test spike

# Figure D

firststim = 3000 # first spike (after 300 ms)
stimfreq = 140  # 140 Hz as in Neubrandt et al.
nrAPs = 15  # as in Neubrandt et al.

# Leads a spike time vector with 15 APs within 100 ms
stimtimes = np.array(get_ISIvec(stimfreq, nrAPs)).cumsum()/dt + firststim
spktr[stimtimes.astype(int)] = 1
t_burstend = np.max(stimtimes).astype(int)

Trange = np.arange(1500, 125000, 5000)  # Range of control spikes
ratio = np.zeros(Trange.shape)
ratio_burstmax = np.zeros(Trange.shape)

for testspike in Trange:
    train = spktr.copy()
    train[[t_burstend + testspike]] = 1
    res = lateSTF.run(train)['nl_readout']
    index = np.where(Trange == testspike)[0]
    uctl = res[0]  # CONTROL SPIKE = BASELINE EFFICACY
    burstmax = res[:t_burstend].max()
    ulate = res[t_burstend + testspike]
    ratio[index] = ulate / uctl
    ratio_burstmax[index] = ulate / burstmax

res_burstonly = lateSTF.run(spktr_burstonly)
res_example = lateSTF.run(spktr_example)

# RUN FOR DIFFERENT NUMBER OF SPIKES (FIGURE E)
# Fix test spike @ 3s after burst
# Run for 0-20 test spikes
# For different burst frequencies (20,40,80,160)

frequencies = [20, 40, 80, 160]
freq_results = dict()
test_delay = int(3000/dt)  # 3s in timesteps
AP_xaxis = np.arange(0, 21)

for f in frequencies:
    spktrs = np.zeros([21, t.shape[0]])
    AP_ratio = np.zeros(np.size(AP_xaxis))
    for aps in range(21):
        spiketimes = np.array(get_ISIvec(f, aps)).cumsum() / dt + firststim
        spiketimes = spiketimes.astype(int)
        if aps > 0:
            spktrs[aps, spiketimes] = 1
            test_time = np.max(spiketimes) + test_delay
        else:
            test_time = test_delay
        spktrs[aps, test_time] = 1

        res = lateSTF.run(spktrs[aps])['nl_readout']
        AP_ratio[aps] = res[test_time] / res[0]

    freq_results[f] = AP_ratio

# PLOTTING FUNCTIONS
# # # # # # # # # #

def plot_data(ax):
    (_, caplines, _) = ax.errorbar(data_x, data_y, yerr=data_yerr, xerr=data_xerr,
                                   capsize=capsize, marker='s', lw=lw, elinewidth=lw * 0.7, markersize=markersize,
                                   color=c_data)
    for capline in caplines:
        capline.set_markeredgewidth(lw * 0.7)
    ax.axhline(y=1, c=c_baseline, ls='dashed', lw=lw*0.6)

    ax.set_xlim(left=-0.5, right=t_modelres)
    ax.set_ylim(bottom=0, top=4.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel('time after burst (s)')
    ax.set_ylabel('test/control EPSC')
    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    ax.yaxis.set_ticks([0, 1, 2, 3, 4])

    return ax


def plot_modelres(ax, ax2):
    x_ax = Trange / 1000 * 0.1
    ax.axhline(y=burstmax / uctl, c=c_burst, ls='dashed', lw=lw*0.6)
    ax.axhline(y=1, c=c_baseline, ls='dashed', lw=lw*0.6)
    ax.text(14.2, burstmax / uctl + 0.2, 'burst\nmaximum', color=c_burst, fontsize=5,
            verticalalignment='bottom', horizontalalignment='right')
    ax.text(14.2, 0.8, 'control', fontsize = 5, color=c_baseline,
            verticalalignment='top', horizontalalignment='right')

    ax.plot(x_ax, ratio, c=c_model, lw=lw)
    ax.set_xlim(left=-0.5, right=t_modelres)
    ax.set_ylim(bottom=0, top=4.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel('time after burst (s)')
    ax.set_ylabel('test/control EPSC')

    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    ax.yaxis.set_ticks([0, 1, 2, 3, 4])

    #for f in frequencies:
    #    ax2.plot(AP_xaxis, freq_results[f], lw=lw)
    ax2.axhline(y=1, c=c_baseline, ls='dashed', lw=lw*0.6)
    ax2.set_ylim(bottom=0, top=4.2)
    ax2.plot(AP_xaxis, freq_results[160], lw=lw, c=c_model)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(left=True, bottom=True)
    ax2.set_xlabel('number of APs in burst')
    ax2.set_ylabel('test/control EPSC')

    return ax, ax2


def plot_gaussians(ax):
    # downsampling data for plotting
    x_ax = (np.arange(t_gaussians) * 0.0001)[::500]
    ax.plot(x_ax, ktot[:t_gaussians:500], c='black', lw=lw, ls='dashed')

    # ax.plot(k1[:t_gaussians], c=c_gaussians[0], lw=lw, zorder=3)
    ax.fill_between(x_ax, 0, k1[:t_gaussians:500], facecolor=c_gaussians[0], alpha=0.6, zorder=3)

    # ax.plot(k2[:t_gaussians], c=c_gaussians[1], lw=lw, zorder=2)
    ax.fill_between(x_ax, 0, k2[:t_gaussians:500], facecolor=c_gaussians[1], alpha=0.6, zorder=2)

    # ax.plot(k3[:t_gaussians], c=c_gaussians[2], lw=lw, zorder=1)
    ax.fill_between(x_ax, 0, k3[:t_gaussians:500], facecolor=c_gaussians[2], alpha=0.6, zorder=1)

    ax.set_title('Efficacy kernel $\mathbf{k}_\mu$', loc='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, bottom=True)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.axis('off')

    return ax


def plot_modelsteps(axes):
    for i in axes[:-1]:
        i.axis('off')
        i.xaxis.set_ticks([])

    x_ax = (np.arange(t_gaussians) * 0.0001)[::100]
    (markers, stemlines, baseline) = axes[0].stem(x_ax, spktr_example[:t_modelsteps:100])
    plt.setp(markers, marker='', markersize=0, markeredgewidth=0)
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, linestyle="-", color="black", linewidth=lw *0.4)
    axes[0].set_title('Spiketrain $S(t)$', loc='center')
    axes[0].set_ylim(0.1)

    axes[1].plot(x_ax, res_burstonly['filtered_s'][:t_modelsteps:100], c='black', lw=lw)
    axes[1].axhline(y=0, c=c_zeroline, ls='dashed')
    axes[1].set_title(r'$\mathbf{k}_\mu\ast S+b$', loc='center')

    axes[2].plot(x_ax, res_burstonly['nl_readout'][:t_modelsteps:100], c='black', lw=lw)
    axes[2].set_title(r'$\mu(t) = f(\mathbf{k}_\mu\ast S+b)$', loc='center')
    axes[2].axhline(y=0, c=c_zeroline, ls='dashed')

    (markers, stemlines, baseline) = axes[3].stem(x_ax, res_example['efficacy'][:t_modelsteps:100])
    plt.setp(markers, marker='', markersize=0, markeredgewidth=0)
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, linestyle="-", color="black", linewidth=lw *0.4)
    axes[3].set_title(r'$E(t) = \mu(t)S(t)$', loc='center')
    axes[3].spines['right'].set_visible(False)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['left'].set_visible(False)
    axes[3].tick_params(which='both', width=0, left=False, bottom=False)
    axes[3].set_xlabel('time (s)')
    axes[3].xaxis.set_ticks([0, 3, 6, 9, 12])
    axes[3].yaxis.set_ticks([])
    axes[3].set_ylim(0.1)
    return axes


def plot():
    # Make Figure Grid
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, wspace=0.1, hspace=0.1)
    axA = fig.add_subplot(spec[0, 0])
    axB = fig.add_subplot(spec[1, 0])
    axC = fig.add_subplot(spec[0, 1])
    axD = fig.add_subplot(spec[0, 2])
    axE = fig.add_subplot(spec[1, 2])
    subspec1 = spec[0:, 1].subgridspec(ncols=1, nrows=5, wspace=0.0, hspace=0.0)
    axC1 = fig.add_subplot(subspec1[0, 0])
    axC2 = fig.add_subplot(subspec1[1, 0])
    axC3 = fig.add_subplot(subspec1[2, 0])
    axC4 = fig.add_subplot(subspec1[3, 0])
    axC5 = fig.add_subplot(subspec1[4, 0])
    axes_modelsteps = [axC2, axC3, axC4, axC5]

    # Make plots
    axA.axis('off')
    axB = plot_data(axB)
    axC.axis('off')
    axC1 = plot_gaussians(axC1)
    axes_modelsteps = plot_modelsteps(axes_modelsteps)
    axD, axE = plot_modelres(axD, axE)

    add_figure_letters([axA, axB, axC, axD, axE], size=12)

    return fig


if __name__ == '__main__':
    import os, inspect

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    fig.savefig(parent_dir + '/Figures/Fig4_raw.pdf')
    plt.show()

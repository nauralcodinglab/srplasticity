import numpy as np
from srplasticity.srp import DetSRP, GaussianKernel
from srplasticity.tools import get_ISIvec
import string

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOT PARAMETERS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

plt.style.use("spiffy")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rc("font", size=7, family="serif")
plt.rc("text", usetex=True)

capsize = 2
markersize = 3
lw = 1

figsize = (5.25102, 5.25102 * 0.7)  # From LaTeX readout of textwidth

# COLORS
c_gaussians = ("#525252", "#969696", "#cccccc")  # greyscale
c_model = "#cc3311"
c_data = "black"
c_zeroline = "#969696"
c_baseline = "black"
c_burst = "#0077bb"

t_modelsteps = 140000  # number of timesteps to plot in modelsteps plots
t_gaussians = 140000  # number of timesteps to plot in gaussian plot
example_Tafter = 19000  # T_after for example test spike in modelsteps plot
t_modelres = 14  # seconds to plot in model result plot

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# EXPERIMENTAL DATA. Estimated from Neubrandt et. al (J Neurosci, 2018) Figure 1
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dataB_x = np.array([0.1, 0.5, 1.0, 2.3, 3.5, 4.8, 6.0, 7.3, 8.50, 9.50, 12.5])
dataB_y = np.array([2.1, 2.7, 3.4, 3.6, 3.1, 3.3, 2.8, 2.9, 2.25, 1.65, 1.20])
dataB_yerr = np.array([0.4, 0.5, 0.4, 0.4, 0.2, 0.3, 0.2, 0.6, 0.25, 0.3, 0.15])
dataB_xerr = np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.2, 0.5])

dataE_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 11.5, 14.5, 21])
dataE_y = np.array([1.05, 1.55, 2.25, 2.35, 2.6, 2.8, 2.6, 3.1, 3.4, 3.45, 3.8, 3.7])
dataE_yerr = np.array([0.1, 0.2, 0.25, 0.4, 0.7, 0.7, 0.3, 0.8, 0.5, 0.3, 0.9, 0.75])
dataE_xerr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 1])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# MODEL PARAMETERS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dt = 0.1  # ms per bin
T = 20e3
t = np.arange(0, T, dt)
L = len(t)

b = -1
mus = [1e3, 2.5e3, 6.0e3]
sigmas = [0.6e3, 1.3e3, 2.8e3]
amps = np.array([125, 620, 1300])

# INITIALIZE MODEL AND KERNELS
kernel = GaussianKernel(amps, mus, sigmas, T, dt)
k1, k2, k3 = kernel._all_gaussians
lateSTF = DetSRP(kernel, b)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# RUN MODEL: FIGURE D
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

spktr_burstonly = np.zeros(L)
spktr_burstonly[
    [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
] = 1  # BURST ONLY TRAIN
spktr_example = spktr_burstonly.copy()
spktr_example[[12000 + example_Tafter]] = 1  # Burst + One Test spike

stimfreq = 140  # 140 Hz as in Neubrandt et al.
nrAPs = 15  # as in Neubrandt et al.

Trange = np.arange(150, 12500, 500)  # Range of control spikes
ratio = np.zeros(Trange.shape)
ratio_burstmax = np.zeros(Trange.shape)

for index, testISI in enumerate(Trange):

    # get spiketimes
    ISIvec = get_ISIvec(stimfreq, nrAPs) + [testISI]

    # Run model
    res = lateSTF.run_ISIvec(ISIvec)[1]

    # extract test amplitude and burstmax
    burstmax = res[:-1].max()
    ratio[index] = res[-1]
    ratio_burstmax[index] = res[-1] / burstmax

res_burstonly = lateSTF.run_spiketrain(spktr_burstonly, return_all=True)
res_example = lateSTF.run_spiketrain(spktr_example, return_all=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# RUN MODEL: FIGURE E
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Fix test spike @ 3s after burst
# Run for 0-20 test spikes
test_delay = 3000  # delay between burst and test spike
AP_xaxis = np.arange(0, 21)
AP_ratio = np.zeros(np.size(AP_xaxis))

for aps in range(21):
    ISIvec = get_ISIvec(stimfreq, aps) + [test_delay]
    res = lateSTF.run_ISIvec(ISIvec)[1]
    AP_ratio[aps] = res[-1]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def add_figure_letters(axes, size=12):
    """
    Function to add Letters enumerating multipanel figures.

    :param axes: list of matplotlib Axis objects to enumerate
    :param size: Font size
    """

    for n, ax in enumerate(axes):

        ax.text(
            -0.15,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=size,
            weight="bold",
            usetex=False,
            family="sans-serif",
        )


def plot_dataB(ax):
    (_, caplines, _) = ax.errorbar(
        dataB_x,
        dataB_y,
        yerr=dataB_yerr,
        xerr=dataB_xerr,
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_data,
    )
    for capline in caplines:
        capline.set_markeredgewidth(lw * 0.7)
    ax.axhline(y=1, c=c_baseline, ls="dashed", lw=lw * 0.6)

    ax.set_xlim(left=-0.5, right=t_modelres)
    ax.set_ylim(bottom=0, top=4.2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel("time after burst (s)")
    ax.set_ylabel("test/control EPSC")
    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    ax.yaxis.set_ticks([0, 1, 2, 3, 4])

    return ax


def plot_dataE(ax):
    (_, caplines, _) = ax.errorbar(
        dataE_x,
        dataE_y,
        yerr=dataE_yerr,
        xerr=dataE_xerr,
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_data,
        label="data",
    )
    for capline in caplines:
        capline.set_markeredgewidth(lw * 0.7)


def plot_modelres(ax, ax2):
    x_ax = Trange / 1000
    ax.axhline(y=burstmax, c=c_burst, ls="dashed", lw=lw * 0.6)
    ax.axhline(y=1, c=c_baseline, ls="dashed", lw=lw * 0.6)
    ax.text(
        14.2,
        burstmax + 0.2,
        "burst\nmaximum",
        color=c_burst,
        fontsize=5,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.text(
        14.2,
        0.8,
        "control",
        fontsize=5,
        color=c_baseline,
        verticalalignment="top",
        horizontalalignment="right",
    )

    ax.plot(x_ax, ratio, c=c_model, lw=lw)
    ax.set_xlim(left=-0.5, right=t_modelres)
    ax.set_ylim(bottom=0, top=4.2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(left=True, bottom=True)
    ax.set_xlabel("time after burst (s)")
    ax.set_ylabel("test/control EPSC")

    ax.xaxis.set_ticks([0, 3, 6, 9, 12])
    ax.yaxis.set_ticks([0, 1, 2, 3, 4])

    # for f in frequencies:
    #    ax2.plot(AP_xaxis, freq_results[f], lw=lw)
    ax2.axhline(y=1, c=c_baseline, ls="dashed", lw=lw * 0.6)
    ax2.set_ylim(bottom=0, top=5.5)
    ax2.plot(AP_xaxis, AP_ratio, lw=lw, c=c_model, zorder=10, label="model")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(left=True, bottom=True)
    ax2.set_xlabel("number of APs in burst")
    ax2.set_ylabel("test/control EPSC")
    ax2.legend(frameon=False, loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5))

    return ax, ax2


def plot_gaussians(ax):
    # downsampling data for plotting
    x_ax = (np.arange(t_gaussians) * 0.0001)[::500]
    ax.plot(x_ax, kernel.kernel[:t_gaussians:500], c="black", lw=lw, ls="dashed")

    ax.fill_between(
        x_ax, 0, k1[:t_gaussians:500], facecolor=c_gaussians[0], alpha=0.6, zorder=3
    )

    ax.fill_between(
        x_ax, 0, k2[:t_gaussians:500], facecolor=c_gaussians[1], alpha=0.6, zorder=2
    )

    ax.fill_between(
        x_ax, 0, k3[:t_gaussians:500], facecolor=c_gaussians[2], alpha=0.6, zorder=1
    )

    ax.set_title("Efficacy kernel $\mathbf{k}_\mu$", loc="center")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, bottom=True)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.axis("off")

    return ax


def plot_modelsteps(axes):
    for i in axes[:-1]:
        i.axis("off")
        i.xaxis.set_ticks([])

    x_ax = (np.arange(t_gaussians) * 0.0001)[::100]
    (markers, stemlines, baseline) = axes[0].stem(
        x_ax, spktr_example[:t_modelsteps:100]
    )
    plt.setp(markers, marker="", markersize=0, markeredgewidth=0)
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, linestyle="-", color="black", linewidth=lw * 0.4)
    axes[0].set_title("Spiketrain $S(t)$", loc="center")
    axes[0].set_ylim(0.1)

    axes[1].plot(
        x_ax, res_burstonly["filtered_spiketrain"][:t_modelsteps:100], c="black", lw=lw
    )
    axes[1].axhline(y=0, c=c_zeroline, ls="dashed")
    axes[1].set_title(r"$\mathbf{k}_\mu\ast S+b$", loc="center")

    axes[2].plot(
        x_ax,
        res_burstonly["nonlinear_readout"][:t_modelsteps:100]
        / res_burstonly["nonlinear_readout"][:t_modelsteps:100][0],
        c="black",
        lw=lw,
    )
    axes[2].set_title(r"$f(b)^{-1} f(\mathbf{k}_\mu\ast S+b)$", loc="center")
    axes[2].axhline(y=0, c=c_zeroline, ls="dashed")

    (markers, stemlines, baseline) = axes[3].stem(
        x_ax, res_example["efficacytrain"][:t_modelsteps:100]
    )
    plt.setp(markers, marker="", markersize=0, markeredgewidth=0)
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, linestyle="-", color="black", linewidth=lw * 0.4)
    axes[3].set_title(r"$E(t)$", loc="center")
    axes[3].spines["right"].set_visible(False)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["left"].set_visible(False)
    axes[3].tick_params(which="both", width=0, left=False, bottom=False)
    axes[3].set_xlabel("time (s)")
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
    axA.axis("off")
    axB = plot_dataB(axB)
    axC.axis("off")
    axC1 = plot_gaussians(axC1)
    axes_modelsteps = plot_modelsteps(axes_modelsteps)
    plot_dataE(axE)
    axD, axE = plot_modelres(axD, axE)

    add_figure_letters([axA, axB, axC, axD, axE], size=10)

    return fig


if __name__ == "__main__":
    import os, inspect

    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    plt.savefig(current_dir + "/figures/Fig4_raw.pdf", bbox_inches="tight")
    plt.show()

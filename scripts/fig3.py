from srplasticity.tm import TsodyksMarkramModel, AdaptedTsodyksMarkramModel
from srplasticity.srp import DetSRP, ExponentialKernel
import numpy as np
import string

import matplotlib
import matplotlib.pyplot as plt
from spiffyplots import MultiPanel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.stats as stats
from srplasticity.tools import get_stimvec

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# PLOT SETTINGS
# # # # # # # # # #
plt.style.use('spiffy')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc("font", size=7, family='serif')
plt.rc('text', usetex=True)

markersize = 3
lw = 1
figsize = (5.25102, 5.25102)  # From LaTeX readout of textwidth

# COLORS
# blue and red (vibrant)
c_13ca = "#cc3311"
c_25ca = "#0077bb"

c_params = ["#000000", "#BBBBBB"]

# PARAMETERS cTM EXAMPLE
# # # # # # # # # # # # #

cTM_13 = TsodyksMarkramModel(
    U=np.array([0.05]),
    f=np.array([0.2]),
    tau_u=np.array([50]),
    tau_r=np.array([10]),
    amp=1
)

cTM_25 = TsodyksMarkramModel(
    U=np.array([0.25]),
    f=np.array([0.2]),
    tau_u=np.array([50]),
    tau_r=np.array([10]),
    amp=1
)

# PARAMETERS aTM EXAMPLE
# # # # # # # # # # # # #

aTM_13 = AdaptedTsodyksMarkramModel(
    U=np.array([0.05]),
    f=np.array([1]),
    tau_u=np.array([50]),
    tau_r=np.array([10]),
    amp=1
)

aTM_25 = AdaptedTsodyksMarkramModel(
    U=np.array([0.25]),
    f=np.array([1]),
    tau_u=np.array([50]),
    tau_r=np.array([10]),
    amp=1
)

# PARAMETERS LNL EXAMPLE
# # # # # # # # # # # # #

srp_13ca = DetSRP(ExponentialKernel(500, 600), -5.5, 1)
srp_25ca = DetSRP(ExponentialKernel(500, 600), -1.5, 1)

# EXAMPLE SPIKES
# # # # # # # # # # # # #
dt = 0.1
nrspikes = 5
isi = 20
isivec = [isi] * nrspikes
examplespikes = get_stimvec([isi] * nrspikes, dt=dt, null=5, extra=30)
t_spiketrain = np.arange(0, len(examplespikes) * dt, dt)

# MODEL FITS
# # # # # # # # # # # # #
fit_cTM13 = cTM_13.run_spiketrain(examplespikes, dt = dt)
fit_cTM25 = cTM_25.run_spiketrain(examplespikes, dt = dt)
fit_aTM13 = aTM_13.run_spiketrain(examplespikes, dt = dt)
fit_aTM25 = aTM_25.run_spiketrain(examplespikes, dt = dt)
fit_srp13 = srp_13ca.run_spiketrain(examplespikes, return_all=True)
fit_srp25 = srp_25ca.run_spiketrain(examplespikes, return_all=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FUNCTIONS
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


def plot():
    # Make Figure Grid
    fig = MultiPanel(grid=[(0, 0), (0, 1), (0, 2),
                           (1, 0), (1, 1), (1, 2),
                           (2, range(0, 2)), (2, 2)],
                     figsize=figsize, wspace=0.1, hspace=0.1)

    fig.panels[6].axis("off")
    subspec1 = fig.gridspec[2, 0:2].subgridspec(ncols=3, nrows=1, wspace=0, hspace=0)

    axC1 = inset_axes(
        fig.panels[6],
        width="100%",
        height="100%",
        loc="upper left",
        bbox_to_anchor=(-0.02, 1, 0.18, 0.3),
        bbox_transform=fig.panels[6].transAxes,
    )
    axC2 = inset_axes(
        fig.panels[6],
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.05, 0.18, 0.6),
        bbox_transform=fig.panels[6].transAxes,
    )

    axC3 = fig.fig.add_subplot(subspec1[0, 1])
    axC4 = fig.fig.add_subplot(subspec1[0, 2])

    # Make plots
    plot_cTM_mech(fig.panels[0], fig.panels[1])
    plot_cTM_eff(fig.panels[2])
    plot_aTM_mech(fig.panels[3], fig.panels[4])
    plot_aTM_eff(fig.panels[5])
    plot_srp(axC1, axC2, axC3, axC4, fig.panels[7])

    add_figure_letters([fig.panels[0], fig.panels[2], fig.panels[3], fig.panels[5], fig.panels[6],
                        fig.panels[7]], size=10)

    return fig


def plot_cTM_mech(ax1, ax2):
    ax1.plot(t_spiketrain, fit_cTM13["u"], color=c_params[0], lw=lw, label="u", zorder=10)
    ax1.plot(t_spiketrain, fit_cTM13["r"], color=c_params[1], lw=lw, label="R")
    ax1.axhline(y=fit_cTM13["u"][0], c=c_13ca, ls="dashed", lw=lw)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=0)
    ax1.set_yticks([fit_cTM13["u"][0], 1])
    ax1.set_yticklabels(["U", 1])
    ax1.get_yticklabels()[0].set_color(c_13ca)
    ax1.legend(frameon=False)
    ax1.set_title('low baseline')

    ax2.plot(t_spiketrain, fit_cTM25["u"], color=c_params[0], lw=lw, label="u", zorder=10)
    ax2.plot(t_spiketrain, fit_cTM25["r"], color=c_params[1], lw=lw, label="R")
    ax2.axhline(y=fit_cTM25["u"][0], c=c_25ca, ls="dashed", lw=lw)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylim(bottom=0, top=1.05)
    ax2.set_xlim(left=0)
    ax2.set_yticks([fit_cTM25["u"][0], 1])
    ax2.set_yticklabels(["U", 1])
    ax2.get_yticklabels()[0].set_color(c_25ca)
    ax2.set_title('high baseline')


def plot_cTM_eff(ax):
    ax.plot(
        np.arange(1, nrspikes + 1),
        fit_cTM13['efficacies'],
        color=c_13ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='low baseline'
    )
    ax.set_xlabel("spike number")

    ax.plot(
        np.arange(1, nrspikes + 1),
        fit_cTM25['efficacies'],
        color=c_25ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='high baseline'
    )
    ax.set_xlabel("spike number")

    ax.xaxis.set_ticks(np.arange(1, 6))
    ax.set_ylim(bottom=0, top=0.7)
    ax.set_ylabel("synaptic efficacy")
    ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6])
    ax.legend(frameon=False)

def plot_aTM_mech(ax1, ax2):
    ax1.plot(t_spiketrain, fit_aTM13["u"], color=c_params[0], lw=lw, label="u", zorder=10)
    ax1.plot(t_spiketrain, fit_aTM13["r"], color=c_params[1], lw=lw, label="R")
    ax1.axhline(y=fit_aTM13["u"][0], c=c_13ca, ls="dashed", lw=lw)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=0)
    ax1.set_yticks([fit_aTM13["u"][0], 1])
    ax1.set_yticklabels(["U", 1])
    ax1.get_yticklabels()[0].set_color(c_13ca)
    ax1.legend(frameon=False)
    ax1.set_title('low baseline')

    ax2.plot(t_spiketrain, fit_aTM25["u"], color=c_params[0], lw=lw, label="u", zorder=10)
    ax2.plot(t_spiketrain, fit_aTM25["r"], color=c_params[1], lw=lw, label="R")
    ax2.axhline(y=fit_aTM25["u"][0], c=c_25ca, ls="dashed", lw=lw)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylim(bottom=0, top=1.05)
    ax2.set_xlim(left=0)
    ax2.set_yticks([fit_aTM25["u"][0], 1])
    ax2.set_yticklabels(["U", 1])
    ax2.get_yticklabels()[0].set_color(c_25ca)
    ax2.set_title('high baseline')


def plot_aTM_eff(ax):

    ax.plot(
        np.arange(1, nrspikes + 1),
        fit_aTM13['efficacies'],
        color=c_13ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='low baseline'
    )
    ax.set_xlabel("spike number")

    ax.plot(
        np.arange(1, nrspikes + 1),
        fit_aTM25['efficacies'],
        color=c_25ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='high baseline'
    )
    ax.set_xlabel("spike number")

    ax.xaxis.set_ticks(np.arange(1, 6))
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("synaptic efficacy")
    ax.yaxis.set_ticks([0, 0.3, 0.6, 0.9])
    ax.legend(frameon=False)

def plot_srp(ax1, ax2, ax3, ax4, ax5):

    kernel = srp_13ca.mu_kernel[0:15000:100]
    t_k = np.arange(0, srp_13ca.mu_kernel.shape[0] * dt, dt) / 1000
    t_k = t_k[0:15000:100]

    ax1.plot(t_k, np.roll(kernel, 1), color="black", lw=lw)
    ax1.set_xlabel("time (s)")
    ax1.set_ylim(0, 1.5)
    ax1.yaxis.set_ticks([])
    ax1.set_xticks([0, 1.5])
    ax1.set_xticklabels([0, 1.5])
    ax1.spines["left"].set_visible(False)
    ax1.text(
        0.75,
        1.5,
        r"$\mathbf{k}_\mu$",
        fontsize=10,
        color="black",
        verticalalignment="top",
        horizontalalignment="center",
    )

    ax2.plot(t_spiketrain, fit_srp13["filtered_spiketrain"] - srp_13ca.mu_baseline, color="black", lw=lw)
    ax2.set_ylim(-0.5, 8)
    ax2.yaxis.set_ticks([0, 5])
    ax2.set_xlabel("time (ms)")
    ax2.text(
        75,
        8,
        r"$\mathbf{k}_\mu\ast S$",
        fontsize=10,
        color="black",
        verticalalignment="top",
        horizontalalignment="center",
    )

    ax3.plot(t_spiketrain, fit_srp13["filtered_spiketrain"], color=c_13ca, lw=lw)
    ax3.plot(t_spiketrain, fit_srp25["filtered_spiketrain"], color=c_25ca, lw=lw)
    ax3.axhline(y=srp_13ca.mu_baseline, c=c_13ca, ls="dashed", lw=lw)
    ax3.axhline(y=srp_25ca.mu_baseline, c=c_25ca, ls="dashed", lw=lw)
    ax3.set_ylim(-6, 6)
    ax3.yaxis.set_ticks([-6, 0, 6])
    ax3.set_xlabel("time (ms)")
    ax3.set_title(r"$\mathbf{k}_\mu\ast S+b$")
    ax3.xaxis.set_ticks([0, 100])

    ax4.plot(t_spiketrain, fit_srp13["nonlinear_readout"], color=c_13ca, lw=lw)
    ax4.plot(t_spiketrain, fit_srp25["nonlinear_readout"], color=c_25ca, lw=lw)
    ax4.set_ylim(bottom=0, top=1)
    ax4.yaxis.set_ticks([0, 0.5, 1])
    ax4.set_yticklabels([0, 0.5, 1])
    ax4.set_xlabel("time (ms)")
    ax4.set_title(r"$f(\mathbf{k}_\mu\ast S+b)$")
    ax4.xaxis.set_ticks([0, 100])

    eff13 = fit_srp13["efficacies"]
    eff25 = fit_srp25["efficacies"]
    ax5.plot(
        np.arange(1, nrspikes + 1),
        eff13,
        color=c_13ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='low baseline'
    )
    ax5.plot(
        np.arange(1, nrspikes + 1),
        eff25,
        color=c_25ca,
        lw=lw,
        marker="o",
        markersize=markersize,
        label='high baseline'
    )
    ax5.set_xlabel("spike number")
    ax5.xaxis.set_ticks(np.arange(1, 6))
    ax5.set_ylim(bottom=0, top=1)
    ax5.set_ylabel("synaptic efficacy")
    ax5.yaxis.set_ticks([0, 0.3, 0.6, 0.9])
    ax5.legend(frameon=False)

    return ax1, ax2, ax3, ax4, ax5


if __name__ == "__main__":
    import os, inspect

    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    plt.savefig(current_dir + "/figures/Fig3_raw.pdf", bbox_inches='tight')
    plt.show()

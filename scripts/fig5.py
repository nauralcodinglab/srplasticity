"""
Code to reproduce Figure 5
"""

from pylab import *
from numpy import *
from scipy.signal import lfilter
from scipy import special
import string
import seaborn as sns

# PLOT SETTINGS
# # # # # # # # # #

plt.style.use("spiffy")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rc("font", size=7, family="serif")
plt.rc("text", usetex=True)

markersize, lw = 3, 1
figsize = (5.25102, 5.25102 * 0.75)  # From LaTeX readout of textwidth

# COLORS
# # # # # # # # # #
c_traces = "lightgrey"
c_green = "#009988"  # green
c_red = "#EE7733"  # red
c_blue = "#0077BB"  # blue


# FUNCTIONS
# # # # # # # # # #


def add_figure_letters(axes, size=14):
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


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + exp(-x))


def exec_func(a, a2, b, b2, tauk):
    """Initialize kernels and find PSC"""
    spktr = zeros(t.shape)
    k = 1 / tauk * exp(-t / tauk)
    spktr[[4000, 4100, 4200, 4300, 4400]] = 1
    filtered_S, filtered_S2 = (
        a * lfilter(k, 1, spktr) + b,
        a2 * lfilter(k, 1, spktr) + b2,
    )
    filtered_S, filtered_S2 = roll(filtered_S, 1), roll(filtered_S2, 1)
    if nlin == "sigmoid":
        mean_om = mufac * sigmoid(filtered_S)
        sig = sigfac * sigmoid(filtered_S2)
    else:
        mean_om = mufac * (tanh(filtered_S) + 1) / 2.0
        sig = sigfac

    I = zeros((Ntrial, len(spktr)))
    for trial in range(Ntrial):
        wspktr, gspktr = mean_om * spktr, sig * spktr
        for ind in where(wspktr > 0)[0]:
            if model == "gamma":
                wspktr[ind] = random.gamma(
                    wspktr[ind] ** 2 / gspktr[ind] ** 2,
                    scale=gspktr[ind] ** 2 / wspktr[ind],
                )
        tau_psc = 5.0
        kappa = -1 * exp(-t / tau_psc)
        I[trial, :] = lfilter(kappa, 1, wspktr) + 0.05 * random.randn(len(spktr))
    return wspktr, I, mean_om, sig, k


def plot_kernels(axA1, axA2):
    """Plot mu and sigma kernels for subplots in panel A"""
    axA1.plot(append(-t[::-1], t), append(k1 * 0 + b, k1 * a + b), "k")
    axA1.set_ylabel("$\mu$-kernel")
    axA1.set_xlabel("time (ms)")
    axA1.set_ylim(bottom=4.2, top=5.2)
    axA1.set_yticks([4.5, 5.0])

    axA2.plot(append(-t[::-1], t), append(k1 * 0 + b2, k1 * a2 + b2), color=c_green)
    axA2.plot(append(-t[::-1], t), append(k2 * 0 + b2, k2 * a3 + b2), "r", color=c_red)
    axA2.plot(append(-t[::-1], t), append(k3 * 0 + b2, k3 * a4 + b2), color=c_blue)
    axA2.set_ylabel("$\sigma$-kernel")
    axA2.set_xlabel("time (ms)")

    return axA1, axA2


def plot_traces(axB):
    """Plot mean PSC and traces"""
    for trial in range(Ntrial):
        axB.plot(t, I1[trial, :], ".6", lw=0.5, color=c_traces)
        axB.plot(t, I2[trial, :] - 15, ".6", lw=0.5, color=c_traces)
        axB.plot(t, I3[trial, :] - 30, ".6", lw=0.5, color=c_traces)
    axB.plot(t, mean(I1, axis=0), "g", lw=1, color=c_green)
    axB.plot(t, mean(I2, axis=0) - 15, "r", lw=1, color=c_red)
    axB.plot(t, mean(I3, axis=0) - 30, "b", lw=1, color=c_blue)

    axB.plot(array([460, 470]), -array([38, 38]), "k", lw=1.5, solid_capstyle="butt")
    axB.set_ylim((-43, 8))
    axB.set_xlim((390, 470))
    axB.axis("off")
    axB.set_ylim(top=10)
    axB.set_xlabel([])

    return axB


def plot_mu_sig(axes1):
    # Plot mean amplitude (panel D) - unaffected by different sigma kernels
    munorm1 = mean_om1[where(wspktr1 > 0)[0][0]]
    axes1[0].plot(
        range(1, 6), mean_om1[where(wspktr1 > 0)[0]] / munorm1, "s-k", ms=3, lw=1
    )

    # Plot panel E; standard from the combination of mu and sigma kernels
    munorm1 = mean_om1[where(wspktr1 > 0)[0][0]]
    munorm2 = mean_om2[where(wspktr2 > 0)[0][0]]
    munorm3 = mean_om3[where(wspktr3 > 0)[0][0]]

    axes1[1].plot(
        range(1, 6),
        sig1[where(wspktr1 > 0)[0]] / munorm1,
        "s-g",
        ms=3,
        lw=1,
        color=c_green,
    )
    axes1[1].plot(
        range(1, 6),
        sig2[where(wspktr2 > 0)[0]] / munorm2,
        "s-r",
        ms=3,
        lw=1,
        color=c_red,
    )
    axes1[1].plot(
        range(1, 6),
        sig3[where(wspktr3 > 0)[0]] / munorm3,
        "s-b",
        ms=3,
        lw=1,
        color=c_blue,
    )

    # Plot panel F; coefficient of variation from the combination of mu and sigma kernels
    axes1[2].plot(
        range(1, 6),
        sig1[where(wspktr1 > 0)[0]] / (mean_om1[where(wspktr1 > 0)[0]]),
        "s-g",
        ms=3,
        lw=1,
        color=c_green,
    )
    axes1[2].plot(
        range(1, 6),
        sig2[where(wspktr2 > 0)[0]] / (mean_om2[where(wspktr2 > 0)[0]]),
        "s-r",
        ms=3,
        lw=1,
        color=c_red,
    )
    axes1[2].plot(
        range(1, 6),
        sig3[where(wspktr3 > 0)[0]] / (mean_om3[where(wspktr3 > 0)[0]]),
        "s-b",
        ms=3,
        lw=1,
        color=c_blue,
    )

    axes1[0].set_ylim((0.88, 1.02))
    axes1[0].set_xticks(range(1, 6))
    axes1[0].set_yticks([0.9, 1.0])
    axes1[0].set_ylabel("mean")
    axes1[0].set_xticklabels([])

    axes1[1].set_ylabel("S.D.")
    axes1[1].set_xticklabels([])
    axes1[1].set_xticks(range(1, 6))

    axes1[2].set_ylim((0.15, 0.68))
    axes1[2].set_xlabel("spike nr.")
    axes1[2].set_ylabel("CV")
    axes1[2].set_xticks(range(1, 6))

    return axes1


def plot_histograms(axes2):
    """Histograms of normalized PSC amplitudes from  first and last stimulation in trains
    axC1 is first stimulation
    axC2 is last stimulation
    """
    # FIRST STIMULATION
    # # # # # # # # # #
    mu1, sigma1 = mean_om1[where(wspktr1 > 0)[0][0]], sig1[where(wspktr1 > 0)[0][0]]
    munorm1 = mean_om1[where(wspktr1 > 0)[0][0]]
    x1 = random.gamma(sigma1, scale=mu1 / sigma1, size=1000)

    mu2, sigma2 = mean_om2[where(wspktr2 > 0)[0][0]], sig2[where(wspktr2 > 0)[0][0]]
    munorm2 = mean_om2[where(wspktr2 > 0)[0][0]]
    x2 = random.gamma(sigma2, scale=mu2 / sigma2, size=1000)

    mu3, sigma3 = mean_om3[where(wspktr3 > 0)[0][0]], sig3[where(wspktr3 > 0)[0][0]]
    munorm3 = mean_om3[where(wspktr3 > 0)[0][0]]
    x3 = random.gamma(sigma3, scale=mu3 / sigma3, size=1000)

    # LAST STIMULATION
    # # # # # # # # # #
    mu4, sigma4 = mean_om1[where(wspktr1 > 0)[0][4]], sig1[where(wspktr1 > 0)[0][4]]
    x4 = random.gamma(sigma4, scale=mu4 / sigma4, size=1000)

    mu5, sigma5 = mean_om2[where(wspktr2 > 0)[0][4]], sig2[where(wspktr2 > 0)[0][4]]
    x5 = random.gamma(sigma5, scale=mu5 / sigma5, size=1000)

    mu6, sigma6 = mean_om3[where(wspktr3 > 0)[0][4]], sig3[where(wspktr3 > 0)[0][4]]
    x6 = random.gamma(sigma6, scale=mu6 / sigma6, size=1000)

    axes2[0].hist(
        x1 / munorm1,
        200,
        range=(0, 20),
        density=True,
        color=c_green,
        histtype=u"step",
        lw=0.75,
    )
    axes2[0].hist(
        x2 / munorm2,
        200,
        range=(0, 20),
        density=True,
        color=c_red,
        histtype=u"step",
        lw=0.75,
    )
    axes2[0].hist(
        x3 / munorm3,
        200,
        range=(0, 20),
        density=True,
        color=c_blue,
        histtype=u"step",
        lw=0.75,
    )

    axes2[1].hist(
        x4 / munorm1,
        200,
        range=(0, 20),
        density=True,
        color=c_green,
        histtype=u"step",
        lw=0.75,
    )
    axes2[1].hist(
        x5 / munorm2,
        200,
        range=(0, 20),
        density=True,
        color=c_red,
        histtype=u"step",
        lw=0.75,
    )
    axes2[1].hist(
        x6 / munorm3,
        200,
        range=(0, 20),
        density=True,
        color=c_blue,
        histtype=u"step",
        lw=0.75,
    )

    for i in range(2):
        axes2[i].set_xlim((0, 6))
        axes2[i].set_xlabel("norm. PSC")
    axes2[0].set_ylabel("prob. density")
    sns.despine()

    return axes2


def plot():
    """Create figure grid and generate plots"""
    fig1 = figure(num=1, figsize=figsize)
    gs = GridSpec(6, 3, wspace=0.5, bottom=0.1, hspace=1)

    axA1 = fig1.add_subplot(gs[0:2, 0])
    axA2 = fig1.add_subplot(gs[0:2, 1])
    axB = fig1.add_subplot(gs[2:4, 0:2])
    axD = fig1.add_subplot(gs[0:2, 2])
    axE = fig1.add_subplot(gs[2:4, 2])
    axF = fig1.add_subplot(gs[4:6, 2])
    axC1 = fig1.add_subplot(gs[4:6, 0])
    axC2 = fig1.add_subplot(gs[4:6, 1])
    axes1 = [axD, axE, axF]
    axes2 = [axC1, axC2]

    # Make plots
    axA1, axA2 = plot_kernels(axA1, axA2)
    axB = plot_traces(axB)
    axes1 = plot_mu_sig(axes1)
    axes2 = plot_histograms(axes2)

    add_figure_letters([axA1, axB, axC1, axD, axE, axF], size=12)

    return fig1


# PARAMETERS
# # # # # # # # # #
Ntrial = 20
dt, T = 0.1, 2e3  # ms per bin, in ms
t = arange(0, T, dt)
nlin = "sigmoid"

b, b2 = 5.0, 1  # baseline for: mu, sigma
a = -300.0  # mu amplitude
a2, a3, a4 = 300.0, 0, -200  # sigma amplitudes
tauk, mufac = 400.0, 5.0
sigfac = 0.45 * mufac * sigmoid(b) / sigmoid(b2)
model = "gamma"
nlin = "sigmoid"

# INITIALIZE KERNELS AND PSC'S
# # # # # # # # # #
wspktr1, I1, mean_om1, sig1, k1 = exec_func(a, a2, b, b2, tauk)
wspktr2, I2, mean_om2, sig2, k2 = exec_func(a, a3, b, b2, tauk)
wspktr3, I3, mean_om3, sig3, k3 = exec_func(a, a4, b, b2, tauk)

if __name__ == "__main__":
    import os, inspect

    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    plt.savefig(current_dir + "/figures/Fig5_raw.pdf")
    plt.show()

from pylab import *
from numpy import *
import sys
import string

from numpy import loadtxt
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy.stats import binom
from scipy.stats import poisson
from scipy import special

import seaborn as sns

# PLOT SETTINGS
# # # # # # # # # #
plt.style.use("science")
matplotlib.rc("xtick", top=False)
matplotlib.rc("ytick", right=False)
matplotlib.rc("ytick.minor", visible=False)
matplotlib.rc("xtick.minor", visible=False)
matplotlib.rc("axes.spines", top=False, right=False)
plt.rc("font", size=8)

# plt.rc('text', usetex=False)
# plt.rc('font', family='sans-serif')

markersize = 3
lw = 1
figsize = (5.25102, 5.25102 * 0.75)  # From LaTeX readout of textwidth

# COLORS
# blue and orange (vibrant)
# c_13ca = '#ee7733'
# c_25ca = '#0077bb'
# blue and red (vibrant)
c_13ca = "#cc3311"
c_13ca_trace = "#f9c7bb"
c_25ca = "#0077bb"
c_25ca_trace = "#bbe6ff"
c_traces = "lightgrey"


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


dt = 0.1  # ms per bin
T = 2e3  # in ms

t = arange(0, T, dt)
nlin = "sigmoid"

b = 5.0
a = -300.0
b2 = 1
a2 = 300.0
tauk = 400.0
mufac = 5.0
gamfac = 0.45 * mufac * sigmoid(b) / sigmoid(b2)
model = "gamma"
Ntrial = 20
nlin = "sigmoid"
execfile("../flexible-STP/ProbabilisticSTP.py")
munorm = mean_om[where(wspktr > 0)[0][0]]

# COLOR GREEN
col = "#009988"

close()
fig1 = figure(num=1, figsize=figsize)
gs = GridSpec(6, 3, wspace=0.5, bottom=0.1, hspace=1)

axA = fig1.add_subplot(gs[0:2, 0])
axA.plot(append(-t[::-1], t), append(k * 0 + b, k * a + b), "k")
axA.set_ylabel("$\mu$-kernel")
axA.set_xlabel("time (ms)")

axB = fig1.add_subplot(gs[0:2, 1])
axB.plot(append(-t[::-1], t), append(k * 0 + b2, k * a2 + b2), color=col)
# axB.set_yticks([-2,0])
axB.set_ylabel("$\sigma$-kernel")
axB.set_xlabel("time (ms)")

axC = fig1.add_subplot(gs[2:4, 0:2])
for trial in range(Ntrial):
    axC.plot(t, I[trial, :], ".6", lw=0.5, color=c_traces)
axC.plot(t, mean(I, axis=0), "g", lw=1, color=col)
axC.set_xlim((390, 470))
axC.plot(array([460, 470]), -array([38, 38]), "k", lw=1.5, solid_capstyle="butt")
axC.axis("off")
axC.set_ylim((-43, 8))

axA1 = fig1.add_subplot(gs[0:2, 2])
axA1.plot(range(1, 6), mean_om[where(wspktr > 0)[0]] / munorm, "s-k", ms=3, lw=1)
axA1.set_ylim((0.88, 1.02))
axA1.set_xticks(range(6))
axA1.set_yticks([0.9, 1.0])
axA1.set_ylabel("mean")
axA1.set_xticklabels([])

axA2 = fig1.add_subplot(gs[4:6, 2])
axA2.plot(
    range(1, 6),
    gam[where(wspktr > 0)[0]] / (mean_om[where(wspktr > 0)[0]]),
    "s-g",
    ms=3,
    lw=1,
    color=col,
)
axA2.set_ylim((0.15, 0.68))
axA2.set_xlabel("spike nr.")
axA2.set_ylabel("CV")
axA2.set_xticks(range(6))
# axA2.set_yticks([0,2])

axA3 = fig1.add_subplot(gs[2:4, 2])
axA3.plot(range(1, 6), gam[where(wspktr > 0)[0]] / munorm, "s-g", ms=3, lw=1, color=col)
# axA3.set_ylim((0,2.01))
axA3.set_ylabel("S.D.")
axA3.set_xticklabels([])
axA3.set_xticks(range(6))
# axA3.set_yticks([])

axF = fig1.add_subplot(gs[4:6, 0])

# x = binom.rvs(int(N),0.5/N,size=1000)+0.05*random.randn(1000)
mu = mean_om[where(wspktr > 0)[0][0]]
gamma = gam[where(wspktr > 0)[0][0]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axF.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)
# axF.set_ylabel('Prob. density')
axF.set_xlim((0, 6))
sns.despine()

axG = fig1.add_subplot(gs[4:6, 1])

mu = mean_om[where(wspktr > 0)[0][4]]
gamma = gam[where(wspktr > 0)[0][4]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axG.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)
axG.set_xlim((0, 6))

axG.set_xlabel("norm. PSC")
axF.set_xlabel("norm. PSC")
axF.set_ylabel("prob. density")
axG.yaxis.set_label_coords(-0.15, 1.0)
sns.despine()

# COLOR RED
col = "#EE7733"

a2 = 0
execfile("../flexible-STP/ProbabilisticSTP.py")
axB.plot(append(-t[::-1], t), append(k * 0 + b2, k * a2 + b2), "r", color=col)
axA2.plot(
    range(1, 6),
    gam[where(wspktr > 0)[0]] / (mean_om[where(wspktr > 0)[0]]),
    "s-r",
    ms=3,
    lw=1,
    color=col,
)
axA3.plot(range(1, 6), gam[where(wspktr > 0)[0]] / munorm, "s-r", ms=3, lw=1, color=col)

# axC = fig1.add_subplot(gs[6, 0])
for trial in range(Ntrial):
    axC.plot(t, I[trial, :] - 15, ".6", lw=0.5, color=c_traces)
axC.plot(t, mean(I, axis=0) - 15, "r", lw=1, color=col)
axC.set_xlim((390, 470))
axC.axis("off")

mu = mean_om[where(wspktr > 0)[0][4]]
gamma = gam[where(wspktr > 0)[0][4]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axG.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)

mu = mean_om[where(wspktr > 0)[0][0]]
gamma = gam[where(wspktr > 0)[0][0]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axF.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)

# COLOR BLUE
col = "#0077BB"

a2 = -200
execfile("../flexible-STP/ProbabilisticSTP.py")
axB.plot(append(-t[::-1], t), append(k * 0 + b2, k * a2 + b2), color=col)
axA2.plot(
    range(1, 6),
    gam[where(wspktr > 0)[0]] / (mean_om[where(wspktr > 0)[0]]),
    "s-b",
    ms=3,
    lw=1,
    color=col,
)
axA3.plot(range(1, 6), gam[where(wspktr > 0)[0]] / munorm, "s-b", ms=3, lw=1, color=col)

for trial in range(Ntrial):
    axC.plot(t, I[trial, :] - 30, ".6", lw=0.5, color=c_traces)
axC.plot(t, mean(I, axis=0) - 30, "b", lw=1, color=col)
axC.set_xlim((390, 470))
axC.axis("off")
axC.set_ylim(top=10)
axC.set_xlabel([])

mu = mean_om[where(wspktr > 0)[0][4]]
gamma = gam[where(wspktr > 0)[0][4]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axG.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)

mu = mean_om[where(wspktr > 0)[0][0]]
gamma = gam[where(wspktr > 0)[0][0]]
x = random.gamma(gamma, scale=mu / gamma, size=1000)
axF.hist(
    x / munorm, 200, range=(0, 20), density=True, color=col, histtype=u"step", lw=0.75
)

axA.set_ylim(bottom=4.2, top=5.2)
axA.set_yticks([4.5, 5.0])

fig1.savefig("../Figures/Fig5-raw.pdf", format="pdf")

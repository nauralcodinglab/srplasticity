import pylab as pl
import numpy as np
from scipy.signal import lfilter
from scipy.stats import binom
from scipy.stats import poisson
from scipy import special

import seaborn as sns

sns.set_style('ticks')
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

# PLOT SETTINGS
# # # # # # # # # #

figsize = (5.0, 3.0)
spiketrain_figsize = (1.5, 0.8)
dpi = 300
linewidth = 0.5


# FUNCTIONS
# # # # # # # # # #

def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

def plot_spiketrain(axZ):
    axZ.plot(t, spktr, 'k', lw=linewidth)
    sns.despine(fig=fig2, ax=axZ, top=True, bottom=True, left=True, right=True)
    axZ.set_xticks([])
    axZ.set_yticks([])
    axZ.set_xticklabels([])
    axZ.set_yticklabels([])
    axZ.plot(np.array([1600, 1800]), np.array([.8, .8]), lw=2, c='k', solid_capstyle='butt')
    axZ.axis('off')

def deterministic_STF(axes_STF):
    axA, axB, axC, axD, axE = axes_STF

    spktr = np.zeros(t.shape)
    tauk = 100.0
    b = -2.0
    a = 700.0
    k = 1 / tauk * np.exp(-t / tauk)
    spktr[[4000, 5000, 6000, 7000, 14000]] = 1
    filtered_S = a * lfilter(k, 1, spktr) + b
    filtered_S = np.roll(filtered_S, 1)
    mean_om = sigmoid(filtered_S)

    tau_psc = 20.0
    kappa = -1 * np.exp(-t / tau_psc)

    axA.set_title('STF')
    axA.plot(t[:10000] - 200, np.roll(k[:10000], 2000), 'k', lw=linewidth)
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_xticklabels([])
    axA.set_yticklabels([])
    sns.despine(fig=fig1, ax=axA, top=True, bottom=True, left=True, right=True)
    axA.set_ylabel('B', rotation=0, visible=True)
    axA.set_ylim((-max(k), max(k)))
    axA.plot(np.array([600, 800]), np.array([.8, .8]) * max(k), lw=2, c='k', solid_capstyle='butt')

    axB.plot(t, filtered_S, 'k', lw=linewidth)
    axB.plot(t, 0 * t, 'r--')
    sns.despine(fig=fig1, ax=axB, top=True, bottom=True, left=True, right=True)
    axB.set_xticks([])
    axB.set_yticks([])
    axB.set_xticklabels([])
    axB.set_yticklabels([])
    axB.set_ylabel('C', rotation=0, visible=True)
    axB.set_ylim((-2.5, 9))
    axB.plot(np.array([1600, 1800]), np.array([4.8, 4.8]), lw=2, c='k', solid_capstyle='butt')

    axC.plot(t, mean_om, 'k', lw=linewidth)
    axC.plot(t[np.where(spktr > 0)[0]], mean_om[np.where(spktr > 0)[0]], 'ok', ms=1, lw=linewidth,
             marker=None)
    sns.despine(fig=fig1, ax=axC, top=True, bottom=True, left=True, right=True)
    axC.set_xticks([])
    axC.set_yticks([])
    axC.set_xticklabels([])
    axC.set_yticklabels([])
    axC.set_ylabel('D', rotation=0, visible=True)

    axD.plot(t, mean_om * spktr, 'k', lw=linewidth)
    sns.despine(fig=fig1, ax=axD, top=True, bottom=True, left=True, right=True)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.set_xticklabels([])
    axD.set_yticklabels([])
    axD.set_ylabel('E', rotation=0, visible=True)

    I = lfilter(kappa, 1, mean_om * spktr) + 0.005 * np.random.randn(len(spktr))
    axE.plot(t, I, 'k', lw=linewidth)
    axE.set_ylim((-1, 0))
    sns.despine(fig=fig1, ax=axE, top=True, bottom=True, left=True, right=True)
    axE.set_xticks([])
    axE.set_yticks([])
    axE.set_xticklabels([])
    axE.set_yticklabels([])
    axE.set_ylabel('F', rotation=0, visible=True)

def deterministic_STD(axes_STD):
    axF, axG, axH, axI, axJ = axes_STD

    spktr = np.zeros(t.shape)
    tauk = 100.0
    b = 2.0
    a = -700.0
    k = 1 / tauk * np.exp(-t / tauk)
    spktr[[4000, 5000, 6000, 7000, 14000]] = 1
    filtered_S = a * lfilter(k, 1, spktr) + b
    filtered_S = np.roll(filtered_S, 1)
    mean_om = sigmoid(filtered_S)

    tau_psc = 20.0
    kappa = -1 * np.exp(-t / tau_psc)

    axF.set_title('STD')
    axF.plot(t[:10000] - 200, np.roll(-k[:10000], 2000), 'k', lw=linewidth)
    axF.set_ylim((-max(k), max(k)))
    axF.axis('off')

    axG.plot(t, filtered_S, 'k', lw=linewidth)
    axG.plot(t, 0 * t, 'r--')
    axG.set_ylim((-9, 2.5))
    axG.axis('off')

    axH.plot(t, mean_om, 'k', lw=linewidth)
    axH.plot(t[np.where(spktr > 0)[0]], mean_om[np.where(spktr > 0)[0]], 'ok', ms=1, marker=None)
    axH.axis('off')

    axI.plot(t, mean_om * spktr, 'k', lw=linewidth)
    axI.axis('off')

    kappa = -1 * np.exp(-t / tau_psc)
    I = lfilter(kappa, 1, mean_om * spktr) + 0.005 * np.random.randn(len(spktr))
    axJ.plot(t, I, 'k', lw=linewidth)
    axJ.set_ylim((-1, 0))
    axJ.axis('off')

def deterministic_STDSTF(axes_STDSTF):
    spktr = np.zeros(t.shape)
    tauk = 100.0
    tau2 = 250.0
    b = -2.0
    a1 = 1000.0
    a2 = -1000.0
    k = 1 / tauk * np.exp(-t / tauk)
    k2 = 1 / tau2 * np.exp(-t / tau2)
    spktr[[4000, 5000, 6000, 7000, 14000]] = 1
    filtered_S = a1 * lfilter(k, 1, spktr) + a2 * lfilter(k2, 1, spktr) + b
    filtered_S = np.roll(filtered_S, 1)
    mean_om = sigmoid(filtered_S)

    tau_psc = 20.0
    kappa = -1 * np.exp(-t / tau_psc)

    axK, axL, axM, axN, axO = axes_STDSTF

    axK.set_title('STF&STD')
    axK.plot(t[:10000] - 200, np.roll(a1 * k[:10000] + a2 * k2[:10000], 2000), 'k', lw=linewidth)
    axK.set_ylim((-max(a1 * k + a2 * k2), max(a1 * k + a2 * k2)))
    axK.axis('off')

    axL.plot(t[1:], filtered_S[1:], 'k', lw=linewidth)
    axL.plot(t[1:], 0 * t[1:], 'r--')
    axL.set_ylim((-4.5, 7))
    axL.axis('off')

    axM.plot(t[1:], mean_om[1:], 'k', lw=linewidth)
    axM.plot(t[np.where(spktr > 0)[0]], mean_om[np.where(spktr > 0)[0]], 'ok', ms=1, marker=None)
    axM.axis('off')

    axN.plot(t, mean_om * spktr, 'k', lw=linewidth)
    axN.axis('off')

    kappa = -1 * np.exp(-t / tau_psc)
    I = lfilter(kappa, 1, mean_om * spktr) + 0.005 * np.random.randn(len(spktr))
    axO.plot(t, I, 'k', lw=linewidth)
    axO.set_ylim((-1, 0))
    axO.axis('off')

def plot():

    axA = fig1.add_subplot(gs[0, 0])
    axB = fig1.add_subplot(gs[1, 0])
    axC = fig1.add_subplot(gs[2, 0])
    axD = fig1.add_subplot(gs[3, 0])
    axE = fig1.add_subplot(gs[4, 0])
    axF = fig1.add_subplot(gs[0, 1])
    axG = fig1.add_subplot(gs[1, 1])
    axH = fig1.add_subplot(gs[2, 1])
    axI = fig1.add_subplot(gs[3, 1])
    axJ = fig1.add_subplot(gs[4, 1])
    axK = fig1.add_subplot(gs[0, 2])
    axL = fig1.add_subplot(gs[1, 2])
    axM = fig1.add_subplot(gs[2, 2])
    axN = fig1.add_subplot(gs[3, 2])
    axO = fig1.add_subplot(gs[4, 2])

    axZ = fig2.add_axes(lim1)

    axes_STF = [axA, axB, axC, axD, axE]
    axes_STD = [axF, axG, axH, axI, axJ]
    axes_STDSTF = [axK, axL, axM, axN, axO]

    plot_spiketrain(axZ)
    deterministic_STD(axes_STD)
    deterministic_STF(axes_STF)
    deterministic_STDSTF(axes_STDSTF)

    return (fig1, fig2)


# SCRIPT
# # # # # # # # # #

# ax1 predefined
dt = 0.1  # ms per bin
T = 2e3  # in ms

t = np.arange(0, T, dt)
spktr = np.zeros(t.shape)
spktr[[4000,5000,6000,7000,14000]] = 1

fig1 = pl.figure(num=1, figsize=figsize, dpi=dpi)
lim1 = [0.24, 0.24, 0.72, 0.70]
gs = pl.GridSpec(5, 3, wspace=0.5, hspace=0.5)

fig2 = pl.figure(num=2, figsize=spiketrain_figsize, dpi=dpi)

if __name__ == '__main__':
    import os, inspect

    # FIGURE 1 PLOTTING

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)

    figure1, figure1_spiketrain = plot()
    figure1.savefig(parent_dir + '/Figures/Fig1.pdf')
    figure1_spiketrain.savefig(parent_dir + '/Figures/Fig1_spiketrain.pdf')

    figure1.show()
    pl.close('all')

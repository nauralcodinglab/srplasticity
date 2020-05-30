# # # # # #
# 'Synaptic Dynamics as Convolutional Units'
#
# Simplified Code for Figure 7
#
# IMPROTANT: This code has been modified to use shorter spike trains and less iterations, to reduce the runtime of the script for
# illustrative abd testing purposes.
# Refer to the Script fig7-simplified.py for the version included in the paper.
# # # # # #

import numpy as np
import math
import matplotlib.pyplot as plt
import random

from pylab import *
from numpy import *
import scipy as sp
import sys

from numpy import loadtxt
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy import special
import seaborn as sns


# PLOT SETTINGS
# # # # # # # # # #
plt.style.use('science')
matplotlib.rc('xtick', top=False)
matplotlib.rc('ytick', right=False)
matplotlib.rc('ytick.minor', visible=False)
matplotlib.rc('xtick.minor', visible=False)
matplotlib.rc('axes.spines', top=False, right=False)
plt.rc('font', size=8)

plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')

markersize = 3
lw = 1
figsize = (5.25102, 5.25102)  # From LaTeX readout of textwidth

import matplotlib.gridspec as gridspec

def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + exp(-x))


def neglogli_old(Y, X, beta1, beta2, gam0, mu0):
    from scipy.special import gamma
    gam = gam0 * sigmoid(dot(beta1, X))
    mu = mu0 * sigmoid(dot(beta2, X))
    nll = 0
    for i in range(Y.shape[1]):
        y = Y[:, i]
        nll = sum(y * gam / mu - (gam - 1) * log(y * gam / mu) + log(mu / gam) + log(gamma(gam)))
    return nll


def neglogli(Y, X, beta1, beta2, sig0, mu0):
    from scipy.special import gamma
    sig = sig0 * sigmoid(dot(beta1, X))
    mu = mu0 * sigmoid(dot(beta2, X))
    nll = 0
    for i in range(Y.shape[1]):
        y = Y[:, i]
        nll = sum(y * (mu / sig ** 2) - ((mu ** 2 / sig ** 2) - 1) * log(y * (mu / sig ** 2)) + log(
            gamma(mu ** 2 / sig ** 2)) + log(sig ** 2 / mu))
    return nll


def sample_old(ntrials, X, beta1, beta2, gam0, mu0):
    gam = gam0 * sigmoid(dot(beta1, X))[0]
    mu = mu0 * sigmoid(dot(beta2, X))[0]
    Y = zeros((len(gam), ntrials))
    for i in range(len(gam)):
        Y[i, :] = random.gamma(gam[i], scale=mu[i] / gam[i], size=ntrials)
    return Y


def sample(ntrials, X, beta1, beta2, sig0, mu0):
    sig = sig0 * sigmoid(dot(beta1, X))[0]
    mu = mu0 * sigmoid(dot(beta2, X))[0]
    Y = zeros((len(sig), ntrials))
    for i in range(len(sig)):
        Y[i, :] = random.gamma((mu[i] ** 2 / sig[i] ** 2), scale=(sig[i] ** 2 / mu[i]), size=ntrials)
    return Y


def initialize():
    beta1 = array([[200.0, -2.0]])
    beta2 = array([[300.0, -2.0]])
    mu0 = 10
    sig0 = 10.
    return beta1, beta2, sig0, mu0


def calc_mse(f, y):
    mse = 0
    for i in range(len(y)):
        mse += (f[i] - y[i]) ** 2
    return mse / len(y)


def spktr(T, dt, tauk, beta):
    t = np.arange(0, T, dt)
    spk = zeros(t.shape)
    k = 1 / tauk * exp(-t / tauk)

    next_spkt = int(np.random.exponential(beta) / dt)
    while next_spkt * dt < T:
        spk[next_spkt] = 1
        next_spkt = next_spkt + int(random.exponential(beta) / dt)
    return spk, k, t


def filtered_trains(spktr, k, ntrials):
    filtered_S = lfilter(k, 1, spktr)
    filtered_S = roll(filtered_S, 1)
    X = array([filtered_S[where(spktr)[0]]])
    X = append(X, ones(X.shape), axis=0)
    beta1, beta2, sig0, mu0 = initialize()
    J2 = sample(ntrials, X, beta1, beta2, sig0, mu0)

    I = zeros((ntrials, len(spktr)))
    for trial in range(ntrials):
        wspktr = spktr * 0
        wspktr[where(spktr > 0)[0]] = J2[:, trial]
        I[trial, :] = lfilter(k, 1, wspktr)

    filtered_S2 = lfilter(k, 1, spktr)
    filtered_S2 = roll(filtered_S2, 1)
    X2 = array([filtered_S2[where(spktr)[0]]])
    X2 = append(X2, ones(X2.shape), axis=0)

    J3 = sample(ntrials, X2, beta1, beta2, sig0, mu0)
    I3 = zeros((ntrials, len(spktr)))
    for trial in range(ntrials):
        wspktr2 = spktr * 0
        wspktr2[where(spktr > 0)[0]] = J3[:, trial]
        I3[trial, :] = lfilter(k, 1, wspktr2)
    return I, I3, J2, X


def trains_mse(spktr, k, Nt, ntrials):
    Imse = []
    for j in range(Nt):
        filtered_S2 = lfilter(k, 1, spktr)
        filtered_S2 = roll(filtered_S2, 1)
        X2 = array([filtered_S2[where(spktr)[0]]])
        X2 = append(X2, ones(X2.shape), axis=0)

        J3 = sample(ntrials, X2, beta1, beta2, sig0, mu0)
        I3 = zeros((ntrials, len(spktr)))
        for trial in range(ntrials):
            wspktr2 = spktr * 0
            wspktr2[where(spktr > 0)[0]] = J3[:, trial]
            I3[trial, :] = lfilter(k, 1, wspktr2)
        Imse.append(np.mean(I3, axis=0))
    return Imse


def filtered_trnsB(spktr, k, ntrials, val):
    filtered_S = lfilter(k, 1, spktr)
    filtered_S = roll(filtered_S, 1)
    X = array([filtered_S[where(spktr)[0]]])
    X = append(X, ones(X.shape), axis=0)
    beta1, beta2, sig0, mu0 = initialize()
    beta1 = array([[200.0, val]])
    J2 = sample(ntrials, X, beta1, beta2, sig0, mu0)
    I = zeros((ntrials, len(spktr)))
    for trial in range(ntrials):
        wspktr = spktr * 0
        wspktr[where(spktr > 0)[0]] = J2[:, trial]
        I[trial, :] = lfilter(k, 1, wspktr)
    return I, J2, X


def search1(J2, X):
    beta1, beta2, sig0, mu0 = initialize()
    param0 = beta1[0, 0]
    param1 = beta2[0, 0]
    prange0 = linspace(param0 * .5, param0 * 1.5, 200)
    prange1 = linspace(param1 * .5, param1 * 1.5, 200)
    P0, P1 = meshgrid(prange0, prange1)
    NLL0 = zeros((len(prange0), len(prange1)))
    for p0 in prange0:
        for p1 in prange1:
            beta1[0, 0] = p0
            beta2[0, 0] = p1
            NLL0[where(prange0 == p0)[0], where(prange1 == p1)[0]] = neglogli(J2, X, beta1, beta2, sig0, mu0)
    cre0, cre1 = param0, param1
    cpr0, cpr1 = prange0[where(NLL0 == NLL0.min())[1][0]], prange1[where(NLL0 == NLL0.min())[0][0]]
    return cre0, cre1, cpr0, cpr1, P0, P1, NLL0, prange0, prange1


def search2(J2, X):
    beta1, beta2, sig0, mu0 = initialize()
    param0 = beta1[0, 1]
    param1 = beta2[0, 1]
    prange0 = linspace(param0 * .5, param0 * 1.5, 200)
    prange1 = linspace(param1 * .5, param1 * 1.5, 200)
    P0, P1 = meshgrid(prange0, prange1)
    NLL1 = zeros((len(prange0), len(prange1)))
    for p0 in prange0:
        for p1 in prange1:
            beta1[0, 1] = p0
            beta2[0, 1] = p1
            NLL1[where(prange0 == p0)[0], where(prange1 == p1)[0]] = neglogli(J2, X, beta1, beta2, sig0, mu0)
    dre0, dre1 = param0, param1
    dpr0, dpr1 = prange0[where(NLL1 == NLL1.min())[1][0]], prange1[where(NLL1 == NLL1.min())[0][0]]
    return dre0, dre1, dpr0, dpr1, P0, P1, NLL1, prange0, prange1


def search3(J2, X):
    beta1, beta2, sig0, mu0 = initialize()
    param1 = beta1[0, 1]
    param0 = sig0
    prange0 = linspace(param0 * .5, param0 * 1.5, 200)
    prange1 = linspace(param1 * .5, param1 * 1.5, 200)
    P0, P1 = meshgrid(prange0, prange1)
    NLL2 = zeros((len(prange0), len(prange1)))
    for p0 in prange0:
        for p1 in prange1:
            beta1[0, 1] = p1
            sig0 = p0
            NLL2[where(prange0 == p0)[0], where(prange1 == p1)[0]] = neglogli(J2, X, beta1, beta2, sig0, mu0)
    ere0, ere1 = param0, param1
    epr0, epr1 = prange0[where(NLL2 == NLL2.min())[1][0]], prange1[where(NLL2 == NLL2.min())[0][0]]
    return ere0, ere1, epr0, epr1, P0, P1, NLL2, prange0, prange1


def search4(J2, X):
    beta1, beta2, sig0, mu0 = initialize()
    param0 = sig0
    param1 = beta1[0, 0]
    prange0 = linspace(param0 * .5, param0 * 1.5, 200)
    prange1 = linspace(param1 * .5, param1 * 1.5, 200)
    P0, P1 = meshgrid(prange0, prange1)
    NLL3 = zeros((len(prange0), len(prange1)))
    for p0 in prange0:
        for p1 in prange1:
            sig0 = p0
            beta1[0, 0] = p1
            NLL3[where(prange0 == p0)[0], where(prange1 == p1)[0]] = neglogli(J2, X, beta1, beta2, sig0, mu0)
    fre0, fre1 = param0, param1
    fpr0, fpr1 = prange0[where(NLL3 == NLL3.min())[1][0]], prange1[where(NLL3 == NLL3.min())[0][0]]
    return fre0, fre1, fpr0, fpr1, P0, P1, NLL3, prange0, prange1


def param_error(Nt, T, dt, tauk, beta, ntrials, val, baseline=False):
    c0err, c1err = np.zeros(Nt), np.zeros(Nt)
    d0err, d1err = np.zeros(Nt), np.zeros(Nt)
    e0err, e1err = np.zeros(Nt), np.zeros(Nt)
    num = 0
    spk, k, t = spktr(T, dt, tauk, beta)
    for i in range(len(spk)):
        if spk[i] == 1:
            num += 1
    for j in range(Nt):
        if baseline == False:
            I, I3, J2, X = filtered_trains(spk, k, ntrials)
            cre0, cre1, cpr0, cpr1, P0, P1, NLL3, prange0, prange1 = search1(J2, X)
            dre0, dre1, dpr0, dpr1, P0, P1, NLL3, prange0, prange1 = search2(J2, X)
            ere0, ere1, epr0, epr1, P0, P1, NLL3, prange0, prange1 = search3(J2, X)
            c0err[j] = abs(cre0 - cpr0) / cre0
            c1err[j] = abs(cre1 - cpr1) / cre1
            d0err[j] = abs(dre0 - dpr0) / dre0
            d1err[j] = abs(dre1 - dpr1) / dre1
            e0err[j] = abs(ere0 - epr0) / ere0
            e1err[j] = abs(ere1 - epr1) / ere1
        else:
            I, J2, X = filtered_trnsB(spk, k, ntrials, val)
            cre0, cre1, cpr0, cpr1, P0, P1, NLL3, prange0, prange1 = search1(J2, X)
            fre0, fre1, fpr0, fpr1, P0, P1, NLL3, prange0, prange1 = search4(J2, X)
            c0err[j] = abs(cre0 - cpr0) / cre0
            c1err[j] = abs(cre1 - cpr1) / cre1
            e0err[j] = abs(fre0 - fpr0) / fre0
    return c0err, c1err, d0err, d1err, e0err, e1err, num


def mse_plot(T, Nt, ntrials):
    Imse = []
    num = 0
    spk, k, t = spktr(T, dt, tauk, beta)
    for i in range(len(spk)):
        if spk[i] == 1:
            num += 1
    for j in range(Nt):
        filtered_S = lfilter(k, 1, spk)
        filtered_S = roll(filtered_S, 1)
        X = array([filtered_S[where(spk)[0]]])
        X = append(X, ones(X.shape), axis=0)
        beta1, beta2, sig0, mu0 = initialize()
        J2 = sample(ntrials, X, beta1, beta2, sig0, mu0)
        I = zeros((ntrials, len(spk))) + 0.05 * random.randn(len(spk))
        for trial in range(ntrials):
            wspktr = spk * 0
            wspktr[where(spk > 0)[0]] = J2[:, trial]
            I[trial, :] = lfilter(k, 1, wspktr)

        spk, k, t = spktr(T, dt, tauk, beta)
        filtered_S2 = lfilter(k, 1, spk)
        filtered_S2 = roll(filtered_S2, 1)
        X2 = array([filtered_S2[where(spk)[0]]])
        X2 = append(X2, ones(X2.shape), axis=0)

        J3 = sample(ntrials, X2, beta1, beta2, sig0, mu0)
        I3 = zeros((ntrials, len(spk)))
        for trial in range(ntrials):
            wspktr2 = spk * 0
            wspktr2[where(spk > 0)[0]] = J3[:, trial]
            I3[trial, :] = lfilter(k, 1, wspktr2) + 0.05 * random.randn(len(spk))
        mse = calc_mse(mean(I3, axis=0), mean(I, axis=0))
        Imse.append(mse)
    return Imse, num


# Gridspec setup details
figstoplot = 1
nlin = 'sigmoid'
close()
fig1 = figure(num=1, figsize=(8.0, 7.0), facecolor='w', dpi=150, edgecolor='w')
gs = GridSpec(6, 4, wspace=0.7, bottom=.2, hspace=0.7)
axA = fig1.add_subplot(gs[1, :2])
axB = fig1.add_subplot(gs[2, :2])
axZ = fig1.add_subplot(gs[3, :2])

# Define params
tauk = 100.0
beta1, beta2, sig0, mu0 = initialize()
dt = 0.1  # ms per bin
T = 3000  # 100e3 # in ms
ntrials = 20
beta = 100.0  # mean ISI in ms

spk, k, t = spktr(T, dt, tauk, beta)
I, I3, J2, X = filtered_trains(spk, k, ntrials)
# Plot spike train
axA.plot(t, spk, 'k', lw=1, solid_capstyle='butt')
axA.axis('off')
axA.set_xlim((400, 2150))

Imse = trains_mse(spk, k, 3, ntrials)
axB.plot(t, mean(I, axis=0), 'k', lw=1.5, label='True 1')
axB.plot(t, mean(I3, axis=0), color='grey', lw=1.5, label='True 2')
axB.set_xlim((400, 2150))
axB.axis('off')
axZ.plot(t, mean(I, axis=0), 'k', lw=1.5)
axZ.plot(t, mean(Imse, axis=0), 'r', lw=1.5, label='Inferred')
axZ.set_xlim((400, 2150))
axZ.axis('off')
h1, l1 = axB.get_legend_handles_labels()
h2, l2 = axZ.get_legend_handles_labels()
axB.legend(h1 + h2, l1 + l2, loc='best', prop={'size': 9})

axC = fig1.add_subplot(gs[0:2, 2])
axD = fig1.add_subplot(gs[0:2, 3])
axE = fig1.add_subplot(gs[2:4, 2])
axF = fig1.add_subplot(gs[2:4, 3])

cre0, cre1, cpr0, cpr1, P0, P1, NLL0, prange0, prange1 = search1(J2, X)
dre0, dre1, dpr0, dpr1, P2, P3, NLL1, prange2, prange3 = search2(J2, X)
ere0, ere1, epr0, epr1, P4, P5, NLL2, prange4, prange5 = search3(J2, X)
fre0, fre1, fpr0, fpr1, P6, P7, NLL3, prange6, prange7 = search4(J2, X)

sns.despine()
axC.set_xlabel('$\sigma{}$ amplitude')
axC.set_ylabel('$\mu{}$ amplitude')
axC.contour(P0, P1, NLL0)
axC.plot(cre0, cre1, '*k')
axC.plot(prange0[where(NLL0 == NLL0.min())[1][0]], prange1[where(NLL0 == NLL0.min())[0][0]], '*r')

sns.despine()
axD.set_xlabel('$\sigma{}$ baseline')
axD.set_ylabel('$\mu{}$ baseline')
axD.contour(P2, P3, NLL1)
axD.plot(dre0, dre1, '*k')
axD.plot(prange2[where(NLL1 == NLL1.min())[1][0]], prange3[where(NLL1 == NLL1.min())[0][0]], '*r')

sns.despine()
axE.set_xlabel('$\sigma{}$ scale factor')
axE.set_ylabel('$\sigma{}$ baseline')
axE.contour(P4, P5, NLL2)
axE.plot(ere0, ere1, '*k')
axE.plot(prange4[where(NLL2 == NLL2.min())[1][0]], prange5[where(NLL2 == NLL2.min())[0][0]], '*r')

sns.despine()
axF.set_xlabel('$\sigma{}$ scale factor')
axF.set_ylabel('$\sigma{}$ amplitude')
axF.contour(P6, P7, NLL3)
axF.plot(fre0, fre1, '*k')
axF.plot(prange6[where(NLL3 == NLL3.min())[1][0]], prange7[where(NLL3 == NLL3.min())[0][0]], '*r')

c0err_n1, c1err_n1, d0err_n1, d1err_n1, e0err_n1, e1err_n1, num1 = param_error(3, 2000, dt, tauk, beta, ntrials, 3.,
                                                                               baseline=False)
c0err_n2, c1err_n2, d0err_n2, d1err_n2, e0err_n2, e1err_n2, num2 = param_error(3, 5000, dt, tauk, beta, ntrials, 3.,
                                                                               baseline=False)
c0err_n3, c1err_n3, d0err_n3, d1err_n3, e0err_n3, e1err_n3, num3 = param_error(3, 7000, dt, tauk, beta, ntrials, 3.,
                                                                               baseline=False)
Ns = array([num1, num2, num3])

c0err, c1err, d0err, d1err, e0err, e1err, num = param_error(3, 5000, dt, tauk, beta, ntrials, -3., baseline=True)
E1 = array([mean(c0err), mean(c1err), mean(e0err)])
SE1 = array([sp.stats.sem(c0err), sp.stats.sem(c1err), sp.stats.sem(e0err)])
c0err, c1err, d0err, d1err, e0err, e1err, num = param_error(3, 5000, dt, tauk, beta, ntrials, 1., baseline=True)
E2 = array([mean(c0err), mean(c1err), mean(e0err)])
SE2 = array([sp.stats.sem(c0err), sp.stats.sem(c1err), sp.stats.sem(e0err)])
c0err, c1err, d0err, d1err, e0err, e1err, num = param_error(3, 5000, dt, tauk, beta, ntrials, 3., baseline=True)
E3 = array([mean(c0err), mean(c1err), mean(e0err)])
SE3 = array([sp.stats.sem(c0err), sp.stats.sem(c1err), sp.stats.sem(e0err)])

ISE1, n1 = mse_plot(3000, 3, 20)
ISE2, n2 = mse_plot(5000, 3, 20)
ISE3, n3 = mse_plot(7000, 3, 20)
NS2 = array([n1, n2, n3])

# close()
# fig1 = figure(num=1,figsize = (8.0,2.0), facecolor = 'w', dpi = 150, edgecolor = 'w')
# gs = GridSpec(1, 4, wspace=0.5, bottom=.2, hspace=0.7)
axG = fig1.add_subplot(gs[4:, 0])
axG2 = fig1.add_subplot(gs[4:, 1])
axG.errorbar(Ns, 100 * array([mean(c0err_n1), mean(c0err_n2), mean(c0err_n3)]),
             yerr=100 * array([sp.stats.sem(c0err_n1), sp.stats.sem(c0err_n2), sp.stats.sem(c0err_n3)]), fmt='o-',
             label='$a_{\sigma}$', color='#009988')
axG.errorbar(Ns, 100 * array([abs(mean(d0err_n1)), abs(mean(d0err_n2)), abs(mean(d0err_n3))]),
             yerr=100 * array([abs(sp.stats.sem(d0err_n1)), abs(sp.stats.sem(d0err_n2)), abs(sp.stats.sem(d0err_n3))]),
             fmt='o-', label='b$_{\sigma}$', color='#EE7733')
axG.errorbar(Ns, 100 * array([mean(e0err_n1), mean(e0err_n2), mean(e0err_n3)]),
             yerr=100 * array([sp.stats.sem(e0err_n1), sp.stats.sem(e0err_n2), sp.stats.sem(e0err_n3)]), fmt='o-',
             label='$\sigma_0$', color='#0077BB')
axG.legend(loc='best', prop={'size': 8})
axG.set_xlabel("number spikes")
axG.set_xticks([0, 50, 100, 150])
axG.set_ylim(0, 38)
axG.set_yticks([0, 10, 20, 30])
axG.set_ylabel("% error")
sns.despine()

axG2.errorbar(Ns, 100 * array([mean(c1err_n1), mean(c1err_n2), mean(c0err_n3)]),
              yerr=100 * array([sp.stats.sem(c1err_n1), sp.stats.sem(c1err_n2), sp.stats.sem(c0err_n3)]), fmt='o-',
              label='$a_{\mu}$', color='#009988')
axG2.errorbar(Ns, 100 * array([abs(mean(e1err_n1)), abs(mean(e1err_n2)), abs(mean(e1err_n3))]),
              yerr=100 * array([abs(sp.stats.sem(e1err_n1)), abs(sp.stats.sem(e1err_n2)), abs(sp.stats.sem(e1err_n3))]),
              fmt='o-', label='b$_{\mu}$', color='#EE7733')
axG2.legend(loc='best', prop={'size': 8})
axG2.set_xticks([0, 50, 100, 150])
axG2.set_ylim(0, 38)
sns.despine()
axG2.set_xlabel("number spikes")

mu_baseline = array([-3, 1, 3])
axH = fig1.add_subplot(gs[4:, 2])
axH.set_xticks([-3, 0, 3, 6])
axH.set_xlabel("baseline")
axH.errorbar(mu_baseline, 100 * array([E1[0], E2[0], E3[0]]), yerr=100 * array([SE1[0], SE2[0], SE3[0]]), fmt='o-',
             label='$a_{\sigma}$', color='#009988')
axH.errorbar(mu_baseline, 100 * array([E1[1], E2[1], E3[1]]), yerr=100 * array([SE1[1], SE2[1], SE3[1]]), fmt='o-',
             label='$a_{\mu}$', color='#EE7733')
axH.errorbar(mu_baseline, 100 * array([E1[2], E2[2], E3[2]]), yerr=100 * array([SE1[2], SE2[2], SE3[2]]), fmt='o-',
             label='$\sigma_0$', color='#0077BB')
axH.set_ylabel("% error")
axH.legend(loc='best', prop={'size': 6})
sns.despine()

axI = fig1.add_subplot(gs[4:, 3])
axI.errorbar(NS2, array([mean(ISE1), mean(ISE2), mean(ISE3)]),
             yerr=array([sp.stats.sem(ISE1), sp.stats.sem(ISE2), sp.stats.sem(ISE3)]), fmt='ko-')
axI.set_xlabel("number spikes")
axI.set_ylabel("MSE")
axI.set_xticks([0, 50, 100, 150])
# axD.set_yticks([0.005,0.01,0.015])
axI.plot([0, 170], [.003, .003], 'k:')
sns.despine()

plt.savefig('../Figures/Fig7-raw-simplified.pdf', format='pdf')

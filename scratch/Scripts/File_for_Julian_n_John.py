#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import matplotlib.pyplot as plt
import random
from pylab import *
from numpy import *
import sys

sys.path.insert(0, "../../matlab_general/")
from scipy.signal import lfilter
from scipy import special

import seaborn as sns

sns.set(style="ticks")
sns.set_style({"xtick.direction": "out", "ytick.direction": "out"})
import pickle as pl
import scipy.optimize as optimize
import scipy as sp


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + exp(-x))


def sample(ntrials, X, beta1, beta2, sig0, mu0):
    sig = sig0 * sigmoid(dot(beta1, X))[0]
    mu = mu0 * sigmoid(dot(beta2, X))[0]
    # print(len(sig),len(mu))
    Y = zeros((len(sig), ntrials))
    for i in range(len(sig)):
        Y[i, :] = random.gamma(
            (mu[i] ** 2 / sig[i] ** 2), scale=(sig[i] ** 2 / mu[i]), size=ntrials
        )
    return Y


def exponential_kernel_weighted(taus, amps, T, dt=0.1):
    """Get an arbitrary number of exponential decay kernels.
    :param taus: list of floats: exponential decays.
    :param amps: list of floats: amplitudes.
    :param T: length of synaptic kernel in ms.
    :param dt: timestep in ms. defaults to 0.1 ms.
    :return: np.array containing the kernel.
    """
    if np.ndim(taus) == 0:
        taus = np.array([taus])
        amps = np.array([amps])
    else:
        taus = np.array(taus)
        amps = np.array(amps)
    t = np.arange(0, T, dt)
    L = len(t)
    n = np.size(taus)  # number of decays
    kernels = np.zeros((n, L))

    for i in range(n):
        tau = taus[i]
        a = amps[i]
        kernels[i,] += (
            a / tau * np.exp(-t / tau)
        )
    return kernels.sum(0)


def exponential_kernel_weighted_nosum(taus, amps, T, dt=0.1):
    """ Same as exponential_kernel_weighted, but returns un-summed result"""
    if np.ndim(taus) == 0:
        taus = np.array([taus])
        amps = np.array([amps])
    else:
        taus = np.array(taus)
        amps = np.array(amps)

    t = np.arange(0, T, dt)
    L = len(t)
    n = np.size(taus)  # number of decays
    kernels = np.zeros((n, L))

    for i in range(n):
        tau = taus[i]
        a = amps[i]
        kernels[i,] += (
            a / tau * np.exp(-t / tau)
        )
    return kernels


def negloglikelihood_II(init_params, Y, taus, spktrain, Nstim, inds):
    """Negative loglikelihood function
    Evalutated using optimize.minimize to return amps, baselines etc that minimized NLL
    """
    from scipy.special import gamma

    Ntrial, T, dt = 20, 1000, 0.1
    a_mu, a_sig = a_mu, a_sig = zeros(len(taus) + 1), zeros(len(taus) + 1)
    a_mu = init_params[0 : len(taus) + 1]
    a_sig = init_params[len(taus) + 1 : -1]
    sig0 = 10.0
    K, K2 = (
        exponential_kernel_weighted_nosum(taus, a_mu[1:], T, dt=0.1),
        exponential_kernel_weighted_nosum(taus, a_sig[1:], T, dt=0.1),
    )
    filtered_S, filtered_S2 = zeros(K.shape), zeros(K2.shape)
    filtered_S3, filtered_S4 = zeros(K.shape), zeros(K2.shape)
    filtered_S5, filtered_S6 = zeros(K.shape), zeros(K2.shape)
    filtered_S7, filtered_S8 = zeros(K.shape), zeros(K2.shape)
    filtered_S9, filtered_S10 = zeros(K.shape), zeros(K2.shape)
    tp, tp2 = zeros((len(a_mu) - 1, Nstim)), zeros((len(a_sig) - 1, Nstim))
    tp3, tp4 = zeros((len(a_mu) - 1, Nstim)), zeros((len(a_sig) - 1, Nstim))
    tp5, tp6 = zeros((len(a_mu) - 1, Nstim)), zeros((len(a_sig) - 1, Nstim))
    tp7, tp8 = zeros((len(a_mu) - 1, Nstim)), zeros((len(a_sig) - 1, Nstim))
    tp9, tp10 = zeros((len(a_mu) - 1, Nstim)), zeros((len(a_sig) - 1, Nstim))

    # Convolve each kernel with spiketrain
    for i in range(K.shape[0]):
        filtered_S[i,], filtered_S2[i,] = (
            lfilter(K[i], 1, spktrain[0]) + a_mu[0],
            lfilter(K2[i], 1, spktrain[0]) + a_sig[0],
        )
        filtered_S[i], filtered_S2[i] = roll(filtered_S[i], 1), roll(filtered_S2[i], 1)
        tp[i], tp2[i] = (
            array([filtered_S[i][where(spktrain[0])[0]]]),
            array([filtered_S2[i][where(spktrain[0])[0]]]),
        )

        filtered_S3[i,], filtered_S4[i,] = (
            lfilter(K[i], 1, spktrain[1]) + a_mu[0],
            lfilter(K2[i], 1, spktrain[1]) + a_sig[0],
        )
        filtered_S3[i], filtered_S4[i] = (
            roll(filtered_S3[i], 1),
            roll(filtered_S4[i], 1),
        )
        tp3[i], tp4[i] = (
            array([filtered_S3[i][where(spktrain[1])[0]]]),
            array([filtered_S4[i][where(spktrain[1])[0]]]),
        )

        filtered_S5[i,], filtered_S6[i,] = (
            lfilter(K[i], 1, spktrain[2]) + a_mu[0],
            lfilter(K2[i], 1, spktrain[2]) + a_sig[0],
        )
        filtered_S5[i], filtered_S6[i] = (
            roll(filtered_S5[i], 1),
            roll(filtered_S6[i], 1),
        )
        tp5[i], tp6[i] = (
            array([filtered_S5[i][where(spktrain[2])[0]]]),
            array([filtered_S6[i][where(spktrain[2])[0]]]),
        )

        filtered_S7[i,], filtered_S8[i,] = (
            lfilter(K[i], 1, spktrain[3]) + a_mu[0],
            lfilter(K2[i], 1, spktrain[3]) + a_sig[0],
        )
        filtered_S7[i], filtered_S8[i] = (
            roll(filtered_S7[i], 1),
            roll(filtered_S8[i], 1),
        )
        tp7[i], tp8[i] = (
            array([filtered_S7[i][where(spktrain[3])[0]]]),
            array([filtered_S8[i][where(spktrain[3])[0]]]),
        )

        filtered_S9[i,], filtered_S10[i,] = (
            lfilter(K[i], 1, spktrain[4]) + a_mu[0],
            lfilter(K2[i], 1, spktrain[4]) + a_sig[0],
        )
        filtered_S9[i], filtered_S10[i] = (
            roll(filtered_S9[i], 1),
            roll(filtered_S10[i], 1),
        )
        tp9[i], tp10[i] = (
            array([filtered_S9[i][where(spktrain[4])[0]]]),
            array([filtered_S10[i][where(spktrain[4])[0]]]),
        )

    # Add a vector of ones for basis function
    tp, tp2 = vstack([tp, ones(Nstim)]), vstack([tp2, ones(Nstim)])
    tp3, tp4 = vstack([tp3, ones(Nstim)]), vstack([tp4, ones(Nstim)])
    tp5, tp6 = vstack([tp5, ones(Nstim)]), vstack([tp6, ones(Nstim)])
    tp7, tp8 = vstack([tp7, ones(Nstim)]), vstack([tp8, ones(Nstim)])
    tp9, tp10 = vstack([tp9, ones(Nstim)]), vstack([tp10, ones(Nstim)])
    a_mu = array(a_mu)
    mu_est = [
        sigmoid(dot(a_mu, tp)),
        sigmoid(dot(a_mu, tp3)),
        sigmoid(dot(a_mu, tp5)),
        sigmoid(dot(a_mu, tp7)),
        sigmoid(dot(a_mu, tp9)),
    ]
    sig_est = [
        sig0 * sigmoid(dot(a_sig, tp2)),
        sig0 * sigmoid(dot(a_sig, tp4)),
        sig0 * sigmoid(dot(a_sig, tp6)),
        sig0 * sigmoid(dot(a_sig, tp8)),
        sig0 * sigmoid(dot(a_sig, tp10)),
    ]
    nll = 0
    for i in range(Y.shape[1]):
        y = Y[:, i]
        mu, sig = mu_est[int(inds[i])], sig_est[int(inds[i])]
        nll = sum(
            y * (mu / sig ** 2)
            - ((mu ** 2 / sig ** 2) - 1) * log(y * (mu / sig ** 2))
            + log(gamma(mu ** 2 / sig ** 2))
            + log(sig ** 2 / mu)
        )
    return nll


def init_gen(L, taus, J4, spktrain, Nstim, inds):
    """Random start points for optimization of NLL
    Runs sets of initial values until a set that does not return inf or nan from NLL is returned
    Make sure ranges for amps == # time constants + 1 re:baseline
    """
    for j in range(50):
        a, c = zeros(L), zeros(L)
        # max and min ranges to choose baseline/amps from
        max_a = [-25, 150, 10, 150]
        min_a = [-40, 70, -10, 70]
        for i in range(L):
            a[i] = np.random.randint(min_a[i], max_a[i], 1) / 10
            c[i] = np.random.randint(min_a[i], max_a[i], 1) / 10
        initial_guess = [a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 10.0]
        # Test initial values to ensure no nan or inf in NLL (introduces bug in optimize.minimize otherwise)
        test = negloglikelihood_II(initial_guess, J4.T, taus, spktrain, Nstim, inds)
        if test == inf or math.isnan(test) is True:
            continue
        else:
            break
    return a, c


def spktrains_gen(T, dt):
    """Spike trains for input data; ordered based on index of data arrays"""
    t = arange(0, T, dt)
    spktrain = zeros((5, len(t)))
    for index in range(5):
        if index == 0:  # 100+20
            spktr = zeros(t.shape)
            spktr[[100, 110, 120, 130, 140, 190, 200, 210, 220, 230]] = 1
            spktrain[0] = spktr
        elif index == 1:  # 20+100
            spktr = zeros(t.shape)
            spktr[[100, 150, 200, 250, 300, 310, 320, 330, 340, 350]] = 1
            spktrain[1] = spktr
        elif index == 2:  # 10+ 100
            spktr = zeros(t.shape)
            spktr[[100, 200, 300, 400, 500, 510, 520, 530, 540, 550]] = 1
            spktrain[2] = spktr
        elif index == 3:  # 100
            spktr = zeros(t.shape)
            spktr[[100, 110, 120, 130, 140, 150, 160, 170, 180, 190]] = 1
            spktrain[3] = spktr
        elif index == 4:  # 20
            spktr = zeros(t.shape)
            spktr[[100, 150, 200, 250, 300, 350, 400, 450, 500, 550]] = 1
            spktrain[4] = spktr
    return spktrain


def test_functions(ND, IND):
    """
    short version of NLL minimization code - useful for testing small adjustments
    runtime of optimize.minimize timed -> max iters fixed low
    """
    T, dt = 1000, 0.1
    taus, Nstim = [15, 150, 700], 10
    spktrain = spktrains_gen(T, dt)
    t = arange(0, T, dt)
    J4 = np.ma.array(ND, mask=np.isnan(ND))
    a, c = init_gen(len(taus) + 1, taus, J4, spktrain, Nstim, IND)
    print(a, c)
    initial_guess = [a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 10.0]
    import time

    start_time = time.time()
    R = optimize.minimize(
        negloglikelihood_II,
        initial_guess,
        (J4.T, taus, spktrain, Nstim, inds),
        method="L-BFGS-B",
        bounds=[
            (-5, 5),
            (-5, 20),
            (-10, 20),
            (-5, 30),
            (-5, 5),
            (-10, 20),
            (-10, 20),
            (-10, 30),
            (0, 10),
        ],
        options={"maxiter": 500},
    )
    print(R.x, R.fun)
    print("--- %s seconds ---" % (time.time() - start_time))
    return


def run_minimize(Nt, ND, IND):
    save, snll = [], []
    for i in range(Nt):
        T, dt = 1000, 0.1
        taus, Nstim = [15, 100, 650], 10
        spktrain = spktrains_gen(T, dt)
        t = arange(0, T, dt)
        J4 = np.ma.array(ND, mask=np.isnan(ND))
        a, c = init_gen(len(taus) + 1, taus, J4, spktrain, Nstim, IND)
        initial_guess = [a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 10]
        result = optimize.minimize(
            negloglikelihood_II,
            initial_guess,
            (J4.T, taus, spktrain, Nstim, IND),
            method="L-BFGS-B",
            bounds=[
                (-5, 5),
                (-5, 25),
                (-5, 2),
                (-5, 25),
                (-5, 5),
                (-5, 20),
                (3, 20),
                (2, 20),
                (0, 10),
            ],
            options={"maxiter": 500},
        )
        save.append(result.x)
        snll.append(result.fun)
        print("iter:", i)
    return snll, save


def test_plot(ind, save, spktr, dat_ind, new_dict):
    Ntrial = 20
    model = "gamma"
    T, dt = 1000, 0.1
    t = arange(0, T, dt)
    Nstim = 10
    taus = [15, 100, 650]
    a_mu, a_sig = zeros(len(taus)), zeros(len(taus))
    a_mu = save[ind][0 : len(taus) + 1]
    a_sig = save[ind][len(taus) + 1 : -1]
    sig0 = 10.0  # save[ind][-1]
    print(a_mu, a_sig, sig0)
    K, K2 = (
        exponential_kernel_weighted(taus, a_mu[1:], T, dt=0.1),
        exponential_kernel_weighted(taus, a_sig[1:], T, dt=0.1),
    )
    filtered_S, filtered_S2 = (
        lfilter(K, 1, spktr) + a_mu[0],
        lfilter(K2, 1, spktr) + a_sig[0],
    )
    filtered_S, filtered_S2 = roll(filtered_S, 1), roll(filtered_S2, 1)
    tp, tp2 = (
        array([filtered_S[where(spktr)[0]]]),
        array([filtered_S2[where(spktr)[0]]]),
    )
    tp, tp2 = vstack([tp, ones(Nstim)]), vstack([tp2, ones(Nstim)])
    a_mu = array(a_mu)
    mu = sigmoid(filtered_S)
    sig = sig0 * sigmoid(filtered_S2)
    for trial in range(Ntrial):
        wspktr = mu * spktr
        for ind in where(wspktr > 0)[0]:
            if model == "gamma":
                wspktr[ind] = np.random.gamma(
                    (mu[ind] * spktr[ind]) ** 2 / (sig[ind] * spktr[ind]) ** 2,
                    scale=(sig[ind] * spktr[ind]) ** 2 / (mu[ind] * spktr[ind]),
                )
    est = mu[where(wspktr > 0)[0]] / mu[where(wspktr > 0)[0][0]]
    if dat_ind < 3:
        plt.plot(est[0:6], "o-")
    else:
        plt.plot(est, "o-")
    plt.plot(
        np.nanmean(new_dict[dat_ind][0], axis=0), "o-"
    )  # 10020, 20100, 10100, 100, 20
    plt.show()

    return


def data():
    pick = open("Toth_Data/10020_normalized_by_cell.pkl", "rb")
    new_dict1 = pl.load(pick)
    pick.close()

    pick = open("Toth_Data/20100_normalized_by_cell.pkl", "rb")
    new_dict2 = pl.load(pick)
    pick.close()

    pick = open("Toth_Data/10100_normalized_by_cell.pkl", "rb")
    new_dict3 = pl.load(pick)
    pick.close()

    pick = open("Toth_Data/100_normalized_by_cell.pkl", "rb")
    new_dict4 = pl.load(pick)
    pick.close()

    pick = open("Toth_Data/20_normalized_by_cell.pkl", "rb")
    new_dict5 = pl.load(pick)
    pick.close()

    alpha1, alpha2, alpha3 = empty((180, 4)), empty((299, 4)), empty((200, 4))
    alpha1[:], alpha2[:], alpha3[:] = np.nan, np.nan, np.nan
    test1, test2, test3 = (
        hstack((new_dict1, alpha1)),
        hstack((new_dict2, alpha2)),
        hstack((new_dict3, alpha3)),
    )
    new_dict = array([[test1], [test2], [test3], [new_dict4], [new_dict5]])

    IND = zeros(1544)
    IND[180:479] = 1
    IND[479:679] = 2
    IND[679:1165] = 3
    IND[1165:] = 4
    ND = np.concatenate(
        (new_dict[0][0], new_dict[1][0], new_dict[2][0], new_dict[3][0], new_dict[4][0])
    )

    import random

    c = list(zip(ND, IND))
    random.shuffle(c)
    ND, IND = zip(*c)
    return ND, IND, new_dict


def main():
    ND, IND, new_dict = data()
    snll, save = run_minimize(2, ND, IND)

    for i in range(len(snll)):
        print(snll[i], save[i])
    T, dt = 1000, 0.1
    spktrain = spktrains_gen(T, dt)
    test_plot(1, save, spktrain[4], 4, new_dict)
    return


main()

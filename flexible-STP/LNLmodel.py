import numpy as np
from tools import sigmoid
from scipy.signal import lfilter


# # # # # # # # # # # #
# LINEAR-NONLINEAR MODEL
# # # # # # # # # # # #

class det_synapse(object):
    """
       Implements a deterministic linear-nonlinear synapse model.
       (Currently implements a single synapse only)
    """

    def __init__(self, k, b, dt=0.1,
                 nlin='sigmoid', n=1, w=1):

        """
        :param k: Numpy Array. Synaptic STP kernel.
        :param b: Float. Baseline.
        :param dt: Float. Timestep in ms.

        :param n: number of synapses        #NotImplemented - fixed to 1
        :param w: weight                    #NotImplemented - fixed to 1
        :param nlin: nonlinear function.    #NotImplemented - always defaults to sigmoid.
        """

        self.dt = dt
        self.T = len(k) * dt # Length of Kernel in ms

        self.n = 1  # fixed to 1
        self.w = 1  # fixed to 1
        self.nlin = sigmoid  # fixed to sigmoid function

        self.b = b
        self.k = k

    def run(self, spktr) -> dict:
        """
        Computes synaptic output for a given input spike train.
        :param spktr: np.array of presynaptic spike trains at each time point
        """

        # assert spiketrain.shape[0] == self.n, 'Spike train does not correspond to number of synapses.'

        filtered_S = np.zeros(spktr.shape)
        L = np.min([len(spktr), len(self.k)])

        for j in np.where(spktr == 1)[0]:
            filtered_S[j:] += self.k[:L- j]
        filtered_S += self.b

        filtered_S = np.roll(filtered_S, 1)
        filtered_S[0] = filtered_S[1]
        mean_om = sigmoid(filtered_S)

        return {'filtered_s': filtered_S,
                'nl_readout': mean_om,
                'efficacy': mean_om * spktr}

class det_exp(det_synapse):
    """
       Implements the deterministic linear-nonlinear synapse model with a kernel based on exponential decays.
       (Currently implements a single synapse only)
    """

    def __init__(self, a, tau, b, T, dt=0.1,
                 nlin='sigmoid', n=1, w=1):

        """
        :param a: np.array or float. Kernel amplitude(s).
        :param tau: np.array or float. Exponential decay time constant(s).
        :param b: Float. Baseline.

        :param T: None or Float. Kernel lenght in ms. None type defaults to 5*max(tau)
        :param dt: Timestep in ms.

        :param n: number of synapses        #NotImplemented - fixed to 1
        :param w: weight                    #NotImplemented - fixed to 1
        :param nlin: nonlinear function.    #NotImplemented - always defaults to sigmoid.
        """

        # Make Kernel
        if T is None:
            T = 5*np.max(tau)

        k = exponential_kernel_weighted(amps=a, taus=tau, T=T, dt=dt)
        self.n_exp = np.shape(k)[0] # number of exponential decays

        # Initialize Synapse
        super().__init__(k, b, dt, nlin, n, w)

class det_gaussian(det_synapse):
    """
       Implements the deterministic linear-nonlinear synapse model with a kernel based on the sum of gaussians.
       (Currently implements a single synapse only)
    """

    def __init__(self, acommon, a, mu, sigma, b, T, dt=0.1,
                 nlin='sigmoid', n=1, w=1):

        """
        :param acommon: float. Common amplitude
        :param a: np.array or float. Gaussian amplitude(s).
        :param mu: np.array or float. Gaussian mean(s).
        :param sigma: np.array or Float. Gaussian std deviation(s).
        :param b: Float. Baseline.

        :param T: None or Float. Kernel lenght in ms. None type defaults to 5 * max(sigma) + max(mu)
        :param dt: Timestep in ms.

        :param n: number of synapses        #NotImplemented - fixed to 1
        :param w: weight                    #NotImplemented - fixed to 1
        :param nlin: nonlinear function.    #NotImplemented - always defaults to sigmoid.
        """

        # Check input parameters
        assert np.size(a) == np.size(mu) == np.size(sigma)

        if np.ndim(mu) == 0:
            mu = np.array([mu])
            sigma = np.array([sigma])
            a = np.array([a])
        else:
            mu = np.array(mu)
            sigma = np.array(sigma)
            a = np.array(a)

        # Kernel
        if T is None:
            T = 5 * np.max(sigma) + np.max(mu)
        k = gaussian_kernel(acommon, a, mu, sigma, T, dt)

        # Initialize Synapse
        super().__init__(k, b, dt, nlin, n, w)

    def run(self, spktr) -> dict:
        """
        Computes synaptic output for a given input spike train.
        :param spktr: np.array of presynaptic spike trains at each time point
        """

        # assert spiketrain.shape[0] == self.n, 'Spike train does not correspond to number of synapses.'

        filtered_S = np.zeros(spktr.shape)
        L = np.min([len(spktr), len(self.k)])

        for j in np.where(spktr == 1)[0]:
            filtered_S[j:] += self.k[:L- j]
        filtered_S += self.b

        filtered_S = np.roll(filtered_S, 1)
        mean_om = sigmoid(filtered_S)

        return {'filtered_s': filtered_S,
                'nl_readout': mean_om,
                'efficacy': mean_om * spktr}

class prob_synapse(det_synapse):
    """
       Implements a probabilistic linear-nonlinear synapse model.
       (Currently implements a single synapse only)
    """

    def __init__(self):
        """
        parameter description
        """
        # Initialize Synapse
        super().__init__(k, b, dt, nlin, n, w)

        # MISSING: Initialize variance / mean kernel

    def run(self, spktr) -> dict:
        """
        Computes probabilistic synaptic output for a given input spike train.
        :param spktr: np.array of presynaptic spike trains at each time point
        """
        NotImplemented

def exponential_kernels(taus, T, dt=0.1):
    """
        Get an arbitrary number of exponential decay kernels.

        :param taus: list of floats: exponential decays.
        :param T: length of synaptic kernel in ms.
        :param dt: timestep in ms. defaults to 0.1 ms.

        :return: np.array containing the kernel.
        """

    if np.ndim(taus) == 0:
        taus = np.array([taus])
    else:
        taus = np.array(taus)

    t = np.arange(0, T, dt)
    L = len(t)

    n = np.size(taus)  # number of decays
    kernels = np.zeros((n, L))

    for i in range(n):
        tau = taus[i]

        kernels[i,] += 1 / tau * np.exp(-t / tau)

    return kernels

def exponential_kernel_weighted(taus, amps, T, dt=0.1):
    """
        Get an arbitrary number of exponential decay kernels.

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

        kernels[i,] += a / tau * np.exp(-t / tau)

    return kernels.sum(0)

def gaussian_kernel(acommon, amps, mus, sigmas, T, dt=0.1):
    """
    Get a kernel from a sum of an arbitrary number of normalized gaussians.

    :param acommon: float: common amplitude.
    :param amps: list of floats: amplitudes.
    :param mus: list of floats: means.
    :param sigmas: list or 1: std deviations.
    :param T: length of synaptic kernel in ms.
    :param dt: timestep in ms. defaults to 0.1 ms.

    :return: np.array containing the kernel.
    """

    amps = np.array(amps)
    mus = np.array(mus)
    sigmas = np.array(sigmas)

    t = np.arange(0, T, dt)
    L = len(t)

    assert np.size(amps) == np.size(mus) == np.size(sigmas)
    n = np.size(amps) # number of gaussians
    kernel = np.zeros(L)

    for i in range(n):
        a = amps[i]
        mu = mus[i]
        sig = sigmas[i]

        kernel += a * np.exp(-(t - mu) ** 2 / 2 / sig ** 2) / np.sqrt(2 * np.pi * sig ** 2)

    return kernel * acommon


# TESTING

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # TESTING THE DETERMINISTIC MODEL WITH GAUSSIAN DECAY KERNEL

    # ax1 predefined
    dt = 0.1  # ms per bin
    T = 100e3  # in ms

    t = np.arange(0, T, dt)
    L = len(t)
    spktr = np.zeros(t.shape)

    b = -1.5

    acommon = 0.25
    mus = [2.0e3, 3.0e3, 6.0e3]
    sigmas = [.5e3, 1.2e3, 2.5e3]
    amps = [600, 4000, 8000]

    ktot = gaussian_kernel(acommon, amps, mus, sigmas, T, dt)

    k1 = gaussian_kernel(acommon, np.array([amps[0]]), np.array([mus[0]]), np.array([sigmas[0]]), T, dt)
    k2 = gaussian_kernel(acommon, np.array([amps[1]]), np.array([mus[1]]), np.array([sigmas[1]]), T, dt)
    k3 = gaussian_kernel(acommon, np.array([amps[2]]), np.array([mus[2]]), np.array([sigmas[2]]), T, dt)

    lateSTF = det_gaussian(acommon, amps, mus, sigmas, b, T, dt)

    Tafter = 50000
    spktr[[4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000 + Tafter]] = 1

    res = lateSTF.run(spktr)
    plt.plot(res['efficacy'][:65000])
    plt.show()
    plt.plot(res['nl_readout'][:65000])
    plt.show()

    # TESTING THE DETERMINISTIC MODEL WITH EXPONENTIAL DECAY KERNEL

    dt = 0.1  # ms per bin
    T = 2e3  # in ms

    t = np.arange(0, T, dt)
    spktr = np.zeros(t.shape)
    spktr[[4000, 5000, 6000, 7000, 14000]] = 1

    tauk = 100.0
    b = -2.0
    a = 700.0
    STF_synapse = det_exp(a, tauk, b, T=3e3)
    STF_res = STF_synapse.run(spktr)

    tauk = 100.0
    tau2 = 250.0
    b = -2.0
    a1 = 1000.0
    a2 = -1000.0
    STFSTD_synapse = det_exp([a1, a2], [tauk, tau2], b, T=3e3)
    STFSTD_res = STFSTD_synapse.run(spktr)

    plt.plot(STFSTD_res['nl_readout'])
    plt.plot(STF_res['nl_readout'])
    plt.show()

    plt.plot(STFSTD_res['efficacy'])
    plt.title('STF/STD')
    plt.show()
    plt.plot((STF_res['efficacy']))
    plt.title('STF')
    plt.show()


    # TESTING EXPONENTIAL CONVOLUTION METHODS

    dt = 0.1  # ms per bin
    T = 2e3  # in ms

    t = np.arange(0, T, dt)
    spktr = np.zeros(t.shape)
    spktr[[4000, 5000, 6000, 7000, 14000]] = 1


    taus = np.array([100.0, 250.0])
    b = -2.0
    amps = np.array([1000.0, -1000.0])

    exps = exponential_kernels(taus, T)
    exps_w = exponential_kernel_weighted(taus,amps,T)

    # INDEPENDENTLY
    filtered_S = np.zeros(spktr.shape)
    for i in range(len(exps)):
        filtered_S += amps[i] * lfilter(exps[i,], 1, spktr)
    filtered_S += b
    filtered_S = np.roll(filtered_S, 1)
    test_indep = sigmoid(filtered_S)

    # SUMMED
    filtered_S = np.zeros(spktr.shape)
    L = np.min([len(spktr), len(exps_w)])

    for j in np.where(spktr == 1)[0]:
        filtered_S[j:] += exps_w[:L - j]
    filtered_S += b

    filtered_S = np.roll(filtered_S, 1)
    test_sum = sigmoid(filtered_S)
"""

This module contains classes for the implementation of the SRP model.
- deterministic SRP model
- probabilistic SRP model
- associated synaptic kernel (gaussian and multiexponential)

Copyright (C) 2021 Julian Rossbroich, Daniel Trotter, John Beninger, Richard Naud

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import lfilter
from srplasticity.tools import get_stimvec


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _refactor_gamma_parameters(mu, sigma):
    """
    Refactor gamma parameters from mean / std to shape / scale
    :param mu: mean parameter as given by the SRP model
    :param sigma: standard deviation parameter as given by the SRP model
    :return: shape and scale parameters
    """
    return (mu ** 2 / sigma ** 2), (sigma ** 2 / mu)


def _sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def _convolve_spiketrain_with_kernel(spiketrain, kernel):
    # add 1 timestep to each spiketime, because efficacy increases AFTER a synaptic release)
    spktr = np.roll(spiketrain, 1)
    spktr[0] = 0  # In case last entry of the spiketrain was a spike
    return lfilter(kernel, 1, spktr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# EFFICIENCY KERNELS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EfficiencyKernel(ABC):

    """ Abstract Base class for a synaptic efficacy kernel"""

    def __init__(self, T=None, dt=0.1):

        self.T = T  # Length of the kernel in ms
        self.dt = dt  # timestep
        self.kernel = np.zeros(int(T / dt))

    @abstractmethod
    def _construct_kernel(self, *args):
        pass


class GaussianKernel(EfficiencyKernel):

    """
    An efficacy kernel from a sum of an arbitrary number of normalized gaussians
    """

    def __init__(self, amps, mus, sigmas, T=None, dt=0.1):
        """
        :param amps: list of floats: amplitudes.
        :param mus: list of floats: means.
        :param sigmas: list or 1: std deviations.
        :param T: length of synaptic kernel in ms.
        :param dt: timestep in ms. defaults to 0.1 ms.
        """

        # Check number of gaussians that make up the kernel
        assert (
            np.size(amps) == np.size(mus) == np.size(sigmas)
        ), "Unequal number of parameters"

        # Default T to largest mean + 5x largest std
        if T is None:
            T = np.max(mus) + 5 * np.max(sigmas)

        # Convert to 1D numpy arrays
        amps = np.atleast_1d(amps)
        mus = np.atleast_1d(mus)
        sigmas = np.atleast_1d(sigmas)

        super().__init__(T, dt)

        self._construct_kernel(amps, mus, sigmas)

    def _construct_kernel(self, amps, mus, sigmas):
        """ constructs the efficacy kernel """

        t = np.arange(0, self.T, self.dt)
        L = len(t)
        n = np.size(amps)  # number of gaussians

        self._all_gaussians = np.zeros((n, L))
        self.kernel = np.zeros(L)

        for i in range(n):
            a = amps[i]
            mu = mus[i]
            sig = sigmas[i]

            self._all_gaussians[i, :] = (
                a
                * np.exp(-((t - mu) ** 2) / 2 / sig ** 2)
                / np.sqrt(2 * np.pi * sig ** 2)
            )

        self.kernel = self._all_gaussians.sum(0)


class ExponentialKernel(EfficiencyKernel):

    """
    An efficacy kernel from a sum of an arbitrary number of Exponential decays
    """

    def __init__(self, taus, amps=None, T=None, dt=0.1):
        """
        :param taus: list of floats: exponential decays.
        :param amps: list of floats: amplitudes (optional, defaults to 1)
        :param T: length of synaptic kernel in ms.
        :param dt: timestep in ms. defaults to 0.1 ms.
        """

        if amps is None:
            amps = np.array([1] * np.size(taus))
        else:
            # Check number of exponentials that make up the kernel
            assert np.size(taus) == np.size(amps), "Unequal number of parameters"

        # Convert to 1D numpy arrays
        taus = np.atleast_1d(taus)
        amps = np.atleast_1d(amps)

        # Default T to 10x largest time constant
        if T is None:
            T = 10 * np.max(taus)

        super().__init__(T, dt)

        self._construct_kernel(amps, taus)

    def _construct_kernel(self, amps, taus):
        """ constructs the efficacy kernel """

        t = np.arange(0, self.T, self.dt)
        L = len(t)
        n = np.size(amps)  # number of gaussians

        self._all_exponentials = np.zeros((n, L))
        self.kernel = np.zeros(L)

        for i in range(n):
            tau = taus[i]
            a = amps[i]

            # set amplitude to a/tau to normalize integrals of all kernels
            self._all_exponentials[i, :] = a / tau * np.exp(-t / tau)

        self.kernel = self._all_exponentials.sum(0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# SRP MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DetSRP:
    def __init__(self, mu_kernel, mu_baseline, mu_scale=None, nlin=_sigmoid, dt=0.1):
        """
        Initialization method for the deterministic SRP model.

        :param kernel: Numpy Array or instance of `EfficiencyKernel`. Synaptic STP kernel.
        :param baseline: Float. Baseline parameter
        :param nlin: nonlinear function. defaults to sigmoid function
        """

        self.dt = dt
        self.nlin = nlin
        self.mu_baseline = mu_baseline

        if isinstance(mu_kernel, EfficiencyKernel):
            assert (
                self.dt == mu_kernel.dt
            ), "Timestep of model and efficacy kernel do not match"
            self.mu_kernel = mu_kernel.kernel
        else:
            self.mu_kernel = np.array(mu_kernel)

        # If no mean scaling parameter is given, assume normalized amplitudes
        if mu_scale is None:
            mu_scale = 1 / self.nlin(self.mu_baseline)
        self.mu_scale = mu_scale

    def run_spiketrain(self, spiketrain, return_all=False):

        filtered_spiketrain = self.mu_baseline + _convolve_spiketrain_with_kernel(
            spiketrain, self.mu_kernel
        )
        nonlinear_readout = self.nlin(filtered_spiketrain) * self.mu_scale
        efficacytrain = nonlinear_readout * spiketrain
        efficacies = efficacytrain[np.where(spiketrain == 1)[0]]

        if return_all:
            return {
                "filtered_spiketrain": filtered_spiketrain,
                "nonlinear_readout": nonlinear_readout,
                "efficacytrain": efficacytrain,
                "efficacies": efficacies,
            }

        else:
            return efficacytrain, efficacies

    def run_ISIvec(self, isivec, **kwargs):
        """
        Returns efficacies given a vector of inter-stimulus-intervals.

        :param isivec: ISI vector
        :param kwargs: Keyword arguments to be passed to 'run' and 'get_stimvec'
        :return: return from `run` method
        """

        spiketrain = get_stimvec(isivec, **kwargs)
        return self.run_spiketrain(spiketrain, **kwargs)


class ProbSRP(DetSRP):
    def __init__(
        self,
        mu_kernel,
        mu_baseline,
        sigma_kernel,
        sigma_baseline,
        mu_scale=None,
        sigma_scale=None,
        **kwargs
    ):
        """
        Initialization method for the probabilistic SRP model.

        :param mu_kernel: Numpy Array or instance of `EfficiencyKernel`. Mean kernel.
        :param mu_baseline: Float. Mean Baseline parameter
        :param sigma_kernel: Numpy Array or instance of `EfficiencyKernel`. Variance kernel.
        :param sigma_baseline: Float. Variance Baseline parameter
        :param sigma_scale: Scaling parameter for the variance kernel
        :param **kwargs: Keyword arguments to be passed to constructor method of `DetSRP`
        """

        super().__init__(mu_kernel, mu_baseline, mu_scale, **kwargs)

        # If not provided, set sigma kernel to equal the mean kernel
        if sigma_kernel is None:
            self.sigma_kernel = self.mu_kernel
            self.sigma_baseline = self.mu_baseline
        else:
            if isinstance(sigma_kernel, EfficiencyKernel):
                assert (
                    self.dt == sigma_kernel.dt
                ), "Timestep of model and variance kernel do not match"
                self.sigma_kernel = sigma_kernel.kernel
            else:
                self.sigma_kernel = np.array(sigma_kernel)

            self.sigma_baseline = sigma_baseline

        # If no sigma scaling parameter is given, assume normalized amplitudes
        if sigma_scale is None:
            sigma_scale = 1 / self.nlin(self.sigma_baseline)
        self.sigma_scale = sigma_scale

    def run_spiketrain(self, spiketrain, ntrials=1):

        spiketimes = np.where(spiketrain == 1)[0]
        efficacytrains = np.zeros((ntrials, len(spiketrain)))

        mean = (
            self.nlin(
                self.mu_baseline
                + _convolve_spiketrain_with_kernel(spiketrain, self.mu_kernel)
            )
            * spiketrain
            * self.mu_scale
        )
        sigma = (
            self.nlin(
                self.sigma_baseline
                + _convolve_spiketrain_with_kernel(spiketrain, self.sigma_kernel)
            )
            * spiketrain
            * self.sigma_scale
        )

        # Sampling from gamma distribution
        efficacies = self._sample(mean[spiketimes], sigma[spiketimes], ntrials)
        efficacytrains[:, spiketimes] = efficacies

        return mean[spiketimes], sigma[spiketimes], efficacies, efficacytrains

    def _sample(self, mean, sigma, ntrials):
        """
        Samples `ntrials` response amplitudes from a gamma distribution given mean and sigma
        """

        return np.random.gamma(
            *_refactor_gamma_parameters(mean, sigma),
            size=(ntrials, len(np.atleast_1d(mean))),
        )


class ExpSRP(ProbSRP):
    """
    SRP model in which mu and sigma kernels are parameterized by a set of amplitudes and respective exponential
    decay time constants.

    This implementation of the SRP model is used for statistical inference of parameters and can be integrated
    between spikes for efficient numerical implementation.
    """

    def __init__(
        self,
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale=None,
        sigma_scale=None,
        **kwargs
    ):

        # Convert to at least 1D arrays
        mu_taus = np.atleast_1d(mu_taus)
        mu_amps = np.atleast_1d(mu_amps)
        sigma_taus = np.atleast_1d(sigma_taus)
        sigma_amps = np.atleast_1d(sigma_amps)

        # Construct mu kernel and sigma kernel from amplitudes and taus
        mu_kernel = ExponentialKernel(mu_taus, mu_amps, **kwargs)
        sigma_kernel = ExponentialKernel(sigma_taus, sigma_amps, **kwargs)

        # Construct with kernel objects
        super().__init__(
            mu_kernel, mu_baseline, sigma_kernel, sigma_baseline, mu_scale, sigma_scale
        )

        # Save amps and taus for version that is integrated between spikes
        self._mu_taus = np.array(mu_taus)
        self._sigma_taus = np.array(sigma_taus)

        # normalize amplitudes by time constant to ensure equal integrals of exponentials
        self._mu_amps = np.array(mu_amps) / self._mu_taus
        self._sigma_amps = np.array(sigma_amps) / self._sigma_taus

        # number of exp decays
        self._nexp_mu = len(self._mu_amps)
        self._nexp_sigma = len(self._sigma_amps)

    def run_ISIvec(self, isivec, ntrials=1, fast=True, **kwargs):
        """
        Overrides the `run_ISIvec` method because the SRP model with
        exponential decays can be integrated between spikes,
        therefore speeding up computation in some cases
        (if ISIs are large, i.e. presynaptic spikes are sparse)

        :return: efficacies
        """

        # Fast evaluation (integrate between spikes)
        if fast:

            state_mu = np.zeros(self._nexp_mu)  # assume kernels have decayed to zero
            state_sigma = np.zeros(
                self._nexp_sigma
            )  # assume kernels have decayed to zero

            means = []
            sigmas = []

            for spike, dt in enumerate(isivec):

                if spike > 0:
                    # At the first spike, read out baseline efficacy
                    # At every following spike, integrate over the ISI and then read out efficacy
                    state_mu = (state_mu + self._mu_amps) * np.exp(-dt / self._mu_taus)
                    state_sigma = (state_sigma + self._sigma_amps) * np.exp(
                        -dt / self._sigma_taus
                    )

                # record value at spike
                means.append(state_mu.sum())
                sigmas.append(state_sigma.sum())

            # Apply nonlinear readout
            means = self.nlin(np.array(means) + self.mu_baseline) * self.mu_scale
            sigmas = (
                self.nlin(np.array(sigmas) + self.sigma_baseline) * self.sigma_scale
            )

            # Sample from gamma distribution
            efficacies = self._sample(means, sigmas, ntrials)

            return means, sigmas, efficacies

        # Standard evaluation (convolution of spiketrain with kernel)
        else:
            return super().run_ISIvec(isivec, **kwargs)

    def reset(self):
        pass

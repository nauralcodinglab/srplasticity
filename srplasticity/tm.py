"""
tm.py Module

This module contains an implementation of the Tsodyks-Markram model of short-term plasticity.

- classic TM model
- adapted TM model to capture supralinear facilitation
- method to fit the TM model to data using an exhaustive parameter grid search
 
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

import numpy as np
from scipy.optimize import brute


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS FOR FITTING PROCEDURE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _sse(targets, estimate):
    """

    :param targets: 2D np.array with response amplitudes of shape [n_sweep, n_stimulus]
    :param estimate: 1D np.array with estimated response amplitudes of shape [n_stimulus]
    :return: sum of squared errors
    """
    return np.nansum((targets - estimate) ** 2)


def _mse(targets, estimate):
    """

    :param targets: 2D np.array with response amplitudes of shape [n_sweep, n_stimulus]
    :param estimate: 1D np.array with estimated response amplitudes of shape [n_stimulus]
    :return: sum of squared errors
    """
    return _sse(targets, estimate) / np.count_nonzero(~np.isnan(targets))


def _total_loss(target_dict, estimates_dict):
    """

    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: total sum of squares
    """
    loss = 0
    for key in target_dict.keys():
        loss += _sse(target_dict[key], estimates_dict[key])
    return loss


def _total_loss_equal_protocol_weights(target_dict, estimates_dict):
    """

    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: total sum of squares
    """
    n_protocols = len(target_dict.keys())
    loss = 0
    for key in target_dict.keys():
        loss += _mse(target_dict[key], estimates_dict[key]) * 1 / n_protocols
    return loss


def _objective_function(x, *args):
    """
    Objective function for scipy.optimize.brute gridsearch

    :param x: parameters for TM model
    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # initialize
    target_dict, stimulus_dict, loss = args
    model = TsodyksMarkramModel(*x)

    # compute estimates
    estimates_dict = {}
    for key, ISIvec in stimulus_dict.items():
        estimates_dict[key] = model.run_ISIvec(ISIvec)
        model.reset()

    # return loss
    if loss == "default":
        return _total_loss(target_dict, estimates_dict)

    elif loss == "equal":
        return _total_loss_equal_protocol_weights(target_dict, estimates_dict)

    elif callable(loss):
        return loss(target_dict, estimates_dict)

    else:
        raise ValueError(
            "Invalid loss function. Check the documentation for valid loss values"
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# TSODYKS-MARKRAM MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class TsodyksMarkramModel:
    def __init__(self, U, f, tau_u, tau_r, amp=None):
        """
        Initialization method for the Tsodyks-Markram model

        :param U: baseline efficacy
        :param f: facilitation constant
        :param tau_u: facilitation timescale
        :param tau_r: depression timescale
        :param amp: baseline amplitude
        """

        # if no amplitude is given, normalize EPSC amplitude to baseline
        if amp is None:
            amp = 1 / U

        self.amp = amp
        self.r = 1
        self.U = U

        self.u = U
        self.f = f
        self.tau_u = tau_u
        self.tau_r = tau_r

    @property
    def _efficacy(self):
        """
        :return synaptic efficacy = R * u * A
        """
        return self.r * self.u * self.amp

    def reset(self):
        """
        reset state variables
        """
        self.u = self.U
        self.r = 1

    def _update(self, dt):
        """
        integrated between spikes given inter-spike-interval dt

        :param dt: time since last spike
        """
        self.r = 1 - (1 - self.r * (1 - self.u)) * np.exp(-dt / self.tau_r)
        self.u = self.U + (self.u + self.f * (1 - self.u) - self.U) * np.exp(
            -dt / self.tau_u
        )

    def _update_ode(self, dt, s):
        """
        Numerically integrate ODEs given timestep and boolean spike variable using forward Euler integration.
        Used when input is a binary spike train and the evolution of state variables is recorded at
        every timestep.

        :param dt: timestep
        :param s: spike (1 or 0)
        """
        self.r += (1 - self.r) * dt / self.tau_r - self.u * self.r * s
        self.u += (self.U - self.u) * dt / self.tau_u + self.f * (1 - self.u) * s

    def run_ISIvec(self, ISIvec):
        """
        numerically efficient implementation.
        Given a vector of inter-spike intervals, `u` and `r` are integrated between spikes

        :param ISIvec: vector of inter-spike intervals
        :return: vector of response efficacies
        """

        efficacies = []

        for spike, dt in enumerate(ISIvec):

            if spike > 0:
                # At the first spike, read out baseline efficacy
                # At every following spike, integrate over the ISI and then read out efficacy
                self._update(dt)
            efficacies.append(self._efficacy)

        return np.array(efficacies)

    def run_spiketrain(self, spiketrain, dt=0.1):
        """
        Numerical evaluation of the model at every timestep.
        Used to demonstrate the evolution of state variables `u` and `r`.

        :param spiketrain: binary spiketrain
        :param dt: timestep (defaults to 0.1 ms)

        :return: dictionary of state variables `u` and `r` and vector of efficacies at each spike
        """
        efficacies = []
        u = np.zeros(len(spiketrain))
        r = np.zeros(len(spiketrain))

        for ix, s in enumerate(spiketrain):
            if s == 1:
                efficacies.append(self._efficacy)
            self._update_ode(dt, s)

            u[ix], r[ix] = self.u, self.r

        return {"u": u, "r": r, "efficacies": np.array(efficacies)}


class AdaptedTsodyksMarkramModel(TsodyksMarkramModel):
    """
    Adapted TM model to capture supralinear facilitation.
    The only difference to the classic TM model is in the update of the
    facilitation parameter `u`:

    Classic model:
            u(n+1) = U + (u + f * (1 - u) - U) * exp(-dt / tau_u)
    Adapted model:
            u(n+1) = U + (u + f * (1 - u) * u - U) * exp(-dt / tau_u)

    """

    def _update(self, dt):
        """
        integrated between spikes given inter-spike-interval dt

        :param dt: time since last spike
        """
        self.r = 1 - (1 - self.r * (1 - self.u)) * np.exp(-dt / self.tau_r)
        self.u = self.U + (self.u + self.f * (1 - self.u) * self.u - self.U) * np.exp(
            -dt / self.tau_u
        )

    def _update_ode(self, dt, s):
        """
        Numerically integrate ODEs given timestep and boolean spike variable using forward Euler integration.
        Used when input is a binary spike train and the evolution of state variables is recorded at
        every timestep.

        :param dt: timestep
        :param s: spike (1 or 0)
        """
        self.r += (1 - self.r) * dt / self.tau_r - self.u * self.r * s
        self.u += (self.U - self.u) * dt / self.tau_u + self.f * (
            1 - self.u
        ) * self.u * s


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING PROCEDURE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def fit_tm_model(
    stimulus_dict, target_dict, parameter_ranges, loss="default", **kwargs
):
    """
    Fitting the TM model to data using a brute Grid-search

    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param parameter_ranges: slice objects for parameters
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.optimize.brute
    """

    return brute(
        _objective_function,
        ranges=parameter_ranges,
        args=(target_dict, stimulus_dict, loss),
        finish=None,
        **kwargs
    )

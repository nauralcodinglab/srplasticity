"""
inference.py

Everything related to parameter inference and fitting the model to data
"""

from srplasticity.srp import ExpSRP
import numpy as np
from scipy.special import gamma  # gamma function
from scipy.optimize import minimize

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS FOR FITTING PROCEDURE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _nll(y, mu, sigma):
    """
    Negative Log Likelihood

    :param y: (np.array) set of amplitudes
    :param mu: (np.array) set of means
    :param sigma: (np.array) set of stds
    """

    return np.nansum((
        y * (mu / sigma ** 2)
        - ((mu ** 2 / sigma ** 2) - 1) * np.log(y * (mu / sigma ** 2))
        + np.log(gamma(mu ** 2 / sigma ** 2))
        + np.log(sigma ** 2 / mu)
    ) / np.count_nonzero(~np.isnan(y), 0))


def _total_loss(target_dict, mean_dict, sigma_dict):
    """

    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: total nll across all stimulus protocols
    """
    loss = 0
    for key in target_dict.keys():
        loss += _nll(target_dict[key], mean_dict[key], sigma_dict[key])

    return loss


def _objective_function(x, *args):
    """
    Objective function for scipy.optimize.minimize

    :param x: parameters for SRP model as a list or array:
                [mu_baseline, *mu_amps,
                sigma_baseline, *sigma_amps, sigma_scale]

    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # Unroll arguments
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale = args

    # Initialize model
    model = ExpSRP(*_convert_fitting_params(x, mu_taus, sigma_taus, mu_scale))

    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    # return loss
    return _total_loss(target_dict, mean_dict, sigma_dict)


def _convert_fitting_params(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1 : 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps : 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# MAIN FITTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def fit_srp_model(
    initial_guess,
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    mu_scale=None,
    bounds=None,
    algo="L-BFGS-B",
    **kwargs
):
    """
    Fitting the SRP model to data using scipy.optimize.minimize

    :param initial_guess: list of parameters:
            [mu_baseline, *mu_amps,sigma_baseline, *sigma_amps, sigma_scale]

    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param mu_taus: predefined time constants for mean kernel
    :param sigma_taus: predefined time constants for sigma kernel
    :param mu_scale: mean scale, defaults to None for normalized data
    :param bounds: bounds for parameters

    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize
    """

    optimizer_res = minimize(
        _objective_function,
        x0=initial_guess,
        method=algo,
        bounds=bounds,
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale),
        **kwargs
    )

    params = _convert_fitting_params(optimizer_res["x"], mu_taus, sigma_taus, mu_scale)

    return (params, optimizer_res)

"""
Everything related to parameter inference and fitting the model to data

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
from scipy.special import gamma  # gamma function
from scipy.optimize import minimize
from scipy._lib._util import MapWrapper
from srplasticity.srp import ExpSRP
from srplasticity.tools import MinimizeWrapper

# Multiprocessing
import copyreg
import types

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

    return np.nansum(
        (
            (y * mu) / (sigma ** 2)
            - ((mu ** 2 / sigma ** 2) - 1) * np.log(y * (mu / (sigma ** 2)))
            + np.log(gamma(mu ** 2 / sigma ** 2))
            + np.log(sigma ** 2 / mu)
        )
    )


def _mean_nll(y, mu, sigma):
    """
    Computes the mean NLL

    :param y: (np.array) set of amplitudes
    :param mu: (np.array) set of means
    :param sigma: (np.array) set of stds
    """

    return _nll(y, mu, sigma) / np.count_nonzero(~np.isnan(y))


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


def _total_loss_equal_protocol_weights(target_dict, mean_dict, sigma_dict):
    """

    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: total sum of squares
    """
    n_protocols = len(target_dict.keys())
    loss = 0
    for key in target_dict.keys():
        loss += (
            _mean_nll(target_dict[key], mean_dict[key], sigma_dict[key])
            * 1
            / n_protocols
        )
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
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss = args

    # Initialize model
    model = ExpSRP(*_convert_fitting_params(x, mu_taus, sigma_taus, mu_scale))

    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    # return loss
    if loss == "default":
        return _total_loss(target_dict, mean_dict, sigma_dict)

    elif loss == "equal":
        return _total_loss_equal_protocol_weights(target_dict, mean_dict, sigma_dict)

    elif callable(loss):
        return loss(target_dict, mean_dict, sigma_dict)

    else:
        raise ValueError(
            "Invalid loss function. Check the documentation for valid loss values"
        )


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


def _default_parameter_bounds(mu_taus, sigma_taus):
    """ returns default parameter boundaries for the SRP fitting procedure """
    return [
        (-6, 6),  # mu baseline
        *[(-10 * tau, 10 * tau) for tau in mu_taus],  # mu amps
        (-6, 6),  # sigma baseline
        *[(-10 * tau, 10 * tau) for tau in sigma_taus],  # sigma amps
        (0.001, 100),  # sigma scale
    ]


def _default_parameter_ranges():
    """
    Default parameter ranges.
    :return list of slice objects
    """

    return (
        slice(-3, 3, 0.5),  # mu_baseline
        slice(-2, 2, 0.25),  # taus
    )


def _starts_from_grid(grid, mu_taus, sigma_taus, sigma_scale=None):
    """
    Converts grid of parameter ranges into initializations
    """
    starts = []
    nstarts, ndims = grid.shape

    for i in range(nstarts):
        params = grid[i]

        if ndims == 2:
            # two dimensions: mu_init equals sigma_init, fixed sigma scale
            assert (
                sigma_scale is not None
            ), "Need to supply sigma scale or more parameter ranges"

            start = [
                params[0],
                *(params[1] * np.array(mu_taus)),
                params[0],
                *(params[1] * np.array(sigma_taus)),
                sigma_scale,
            ]

        elif ndims == 3:
            # three dimensions: also change sigma_scale
            start = [
                params[0],
                *(params[1] * np.array(mu_taus)),
                params[0],
                *(params[1] * np.array(sigma_taus)),
                params[2],
            ]

        elif ndims == 5:
            # five dimensions: initializations for sigma independent
            start = [
                params[0],
                *(params[1] * np.array(mu_taus)),
                params[3],
                *(params[4] * np.array(sigma_taus)),
                params[2],
            ]

        else:
            raise ValueError("supply either 2, 3 or 5 ranges of parameters")

        starts.append(start)
    return np.array(starts)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# INITIALIZATION
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _get_grid(ranges, Ns=3):
    # CODE COPIED FROM SCIPY.OPTIMIZE.BRUTE:
    N = len(ranges)

    lrange = list(ranges)
    for k in range(N):
        if type(lrange[k]) is not type(slice(None)):
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            lrange[k] = slice(*lrange[k])
    if N == 1:
        lrange = lrange[0]

    grid = np.mgrid[lrange]

    # obtain an array of parameters that is iterable by a map-like callable
    inpt_shape = grid.shape
    if N > 1:
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    return grid


class RandomDisplacement(object):
    """
    Random displacement of SRP parameters
    Calling this updates `x` in-place.

    Parameters
    ----------
    max_stepsize: np.array: maximum stepsize in each dimension
    """

    def __init__(
        self,
        bounds=False,
        max_stepsize="default",
        mu_taus=None,
        sigma_taus=None,
        disp=True,
    ):

        self.disp = disp
        if max_stepsize == "default":
            self.max_stepsize = np.array(
                [(2, *np.array(mu_taus), 2, *np.array(sigma_taus), 1)]
            )

    def __call__(self, x):
        newx = x + self._sample()
        if self.disp:
            print("New initial guess:")
            print(newx)

        return newx

    def _sample(self):
        return np.random.uniform(-self.max_stepsize, self.max_stepsize)


class Grid(object):
    """
    Grid Search starts for SRP parameters

    Parameters
    ----------
    ranges: parameter ranges (as e.g. passed into scipy.optimize.brute)
        see scipy.optimize documentation on how to pass in clide objects or
        range tuples and Ns
    """

    def __init__(self, ranges, Ns=3):

        self.grid = _get_grid(ranges, Ns)
        self.nstart = 0

    def __call__(self, x):
        newx = self.grid[self.nstart]
        self.nstart += 1
        return newx


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# MAKE THINGS PICKLEABLE FOR MULTIPROCESSING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# MAIN FITTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def fit_srp_model_gridsearch(
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    param_ranges="default",
    mu_scale=None,
    sigma_scale=1,
    bounds="default",
    method="L-BFGS-B",
    loss="default",
    workers=1,
    **kwargs
):
    """
    Fitting the SRP model using a gridsearch.

    :param stimulus_dict: dictionary of protocol key - isivec mapping
    :param target_dict: dictionary of protocol key - target amplitudes
    :param mu_taus: mu time constants
    :param sigma_taus: sigma time constants
    :param target_dict: dictionary of protocol key - target amplitudes
    :param param_ranges: Optional - ranges of parameters in form of a tuple of slice objects
    :param mu_scale: mu scale (defaults to None for normalized data)
    :param sigma_scale: sigma scale in case param_ranges only covers 2 dimensions
    :param bounds: bounds for parameters to be passed to minimizer function
    :param method: algorithm for minimizer function
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param workers: number of processors
    """

    # 1. SET PARAMETER BOUNDS
    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)

    # 2. INITIALIZE WRAPPED MINIMIZER FUNCTION
    wrapped_minimizer = MinimizeWrapper(
        _objective_function,
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss),
        bounds=bounds,
        method=method,
        **kwargs
    )

    # 3. MAKE GRID
    if param_ranges == "default":
        param_ranges = _default_parameter_ranges()
    grid = _get_grid(param_ranges)
    starts = _starts_from_grid(grid, mu_taus, sigma_taus, sigma_scale)

    # 4. RUN

    print("STARTING GRID SEARCH FITTING PROCEDURE")
    print("- Using {} cores in parallel".format(workers))
    print("- Iterating over a total of {} initial starts".format(len(grid)))

    print("Make a coffee. This might take a while...")

    # CODE COPIED FROM SCIPY.OPTIMIZE.BRUTE:
    # iterate over input arrays, possibly in parallel
    with MapWrapper(pool=workers) as mapper:
        listres = np.array(list(mapper(wrapped_minimizer, starts)))

    fval = np.array(
        [res["fun"] if res["success"] is True else np.nan for res in listres]
    )

    bestsol_ix = np.nanargmin(fval)
    bestsol = listres[bestsol_ix]
    bestsol["initial_guess"] = starts[bestsol_ix]

    fitted_params = _convert_fitting_params(bestsol["x"], mu_taus, sigma_taus, mu_scale)

    return fitted_params, bestsol, starts, fval, listres


def fit_srp_model(
    initial_guess,
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    mu_scale=None,
    bounds="default",
    loss="default",
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
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param algo: Algorithm for fitting procedure
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize
    """

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)

    optimizer_res = minimize(
        _objective_function,
        x0=initial_guess,
        method=algo,
        bounds=bounds,
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss),
        **kwargs
    )

    params = _convert_fitting_params(optimizer_res["x"], mu_taus, sigma_taus, mu_scale)

    return params, optimizer_res

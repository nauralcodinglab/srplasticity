"""
inference.py

Everything related to parameter inference and fitting the model to data
"""

from srplasticity.srp import ExpSRP
import numpy as np
from scipy.special import gamma  # gamma function
from scipy.optimize import minimize, basinhopping
from scipy._lib._util import MapWrapper

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
            y * (mu / sigma ** 2)
            - ((mu ** 2 / sigma ** 2) - 1) * np.log(y * (mu / sigma ** 2))
            + np.log(gamma(mu ** 2 / sigma ** 2))
            + np.log(sigma ** 2 / mu)
        )
        / np.count_nonzero(~np.isnan(y), 0)
    )


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


def _default_parameter_bounds(n_mu_taus, n_sigma_taus):
    """ returns default parameter boundaries for the SRP fitting procedure """
    return [
        (-5, 5),  # mu baseline
        *[(-np.inf, np.inf)] * n_mu_taus,  # mu taus
        (-5, 5),  # sigma baseline
        *[(-np.inf, np.inf)] * n_sigma_taus,  # sigma taus
        (0.001, np.inf),  # sigma scale
    ]


class _Minimize_Wrapper(object):
    """
    Object to wrap scipy optimize.minimize function for grid search

    :param func: objective function to call the minimizer on
    :param args: arguments for objective function
    :param kwargs: other keyword arguments for minimizer
    """

    def __init__(self, func, args, **kwargs):
        self.minimizer = minimize
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.minimizer(self.func, x0=x, args=self.args, **self.kwargs)


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
# MAIN FITTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def fit_srp_model_gridsearch(
    stimulus_dict,
    target_dict,
    param_ranges,
    mu_taus,
    sigma_taus,
    mu_scale=None,
    bounds="default",
    workers=1,
    **kwargs
):

    # 1. SET PARAMETER BOUNDS
    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(len(mu_taus), len(sigma_taus))

    # 2. INITIALIZE WRAPPED MINIMIZER FUNCTION
    wrapped_minimizer = _Minimize_Wrapper(_objective_function,
                                          args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale),
                                          bounds=bounds,
                                          **kwargs)

    # 3. MAKE GRID
    N = len(param_ranges)
    grid = _get_grid(param_ranges)
    inpt_shape = grid.shape

    # 4. RUN
    # CODE COPIED FROM SCIPY.OPTIMIZE.BRUTE:
    # iterate over input arrays, possibly in parallel
    with MapWrapper(pool=workers) as mapper:
        Jout = np.array(list(mapper(wrapped_minimizer, grid)))
        if N == 1:
            grid = (grid,)
            Jout = np.squeeze(Jout)
        elif N > 1:
            Jout = np.reshape(Jout, inpt_shape[1:])
            grid = np.reshape(grid.T, inpt_shape)

    Nshape = np.shape(Jout)

    indx = np.argmin(Jout.ravel(), axis=-1)
    Nindx = np.zeros(N, int)
    xmin = np.zeros(N, float)
    for k in range(N - 1, -1, -1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx // thisN
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]

    Jmin = Jout[tuple(Nindx)]
    if (N == 1):
        grid = grid[0]
        xmin = xmin[0]




def fit_srp_model(
    initial_guess,
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    mu_scale=None,
    bounds="default",
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

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(len(mu_taus), len(sigma_taus))

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

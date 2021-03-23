"""
_tools.py Module

This module contains tools and helper functions that are used by other modules
or that are useful for implementations

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
from scipy.optimize import minimize


def get_stimvec(ISIvec, dt=0.1, null=0, extra=10):
    """
    Generates a binary stimulation vector from a vector with ISI intervals
    :param ISIvec: ISI interval vector (in ms)
    :param dt: timestep (ms)
    :param null: 0s in front of the vector (in ms)
    :param extra: 0s after the last stimulus (in ms)
    :return: binary stim vector
    """

    ISIindex = np.cumsum(
        np.round(np.array([i if i == 0 else i - dt for i in ISIvec]) / dt, 1)
    )
    # ISI times accounting for base zero-indexing

    spktr = np.array(
        [0] * int(null / dt)
        + [
            1 if i in ISIindex.astype(int) else 0
            for i in np.arange(int(sum(ISIvec) / dt + extra / dt))
        ]
    ).astype(bool)

    # Remove redundant dimension
    return spktr


def get_ISIvec(freq, nstim):
    """
    Returns an ISI vector of a periodic stimulation train (constant frequency)
    :param freq: int of stimulation frequency
    :param nstim: number of stimuli in the train
    :return: ISI vector in ms
    """
    if nstim == 0:
        return []
    else:
        return [0] + list(np.array([1000 / freq]).astype(int)) * (nstim - 1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# MULTIPROCESSING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class MinimizeWrapper(object):
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

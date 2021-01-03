"""
tools.py Module

This module contains tools and helper functions that are used by other modules
or that are useful for implementations
"""

import numpy as np


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

    return np.array(
        [0] * int(null / dt)
        + [
            1 if i in ISIindex.astype(int) else 0
            for i in np.arange(int(sum(ISIvec) / dt + extra / dt))
        ]
    ).astype(bool)


def get_ISIvec(freq, nstim):
    """
    Returns an ISI vector of a periodic stimulation train (constant frequency)
    :param freq: int of stimulation frequency
    :param nstim: number of stimuli in the train
    :return: ISI vector in ms
    """
    return [0] + list(np.array([1000 / freq]).astype(int)) * (nstim - 1)

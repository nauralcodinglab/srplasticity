import numpy as np
import pickle
import os, inspect
import matplotlib.pyplot as plt
import string
import pandas as pd

def add_figure_letters(axes, size = 14):
    """
    Function to add Letters enumerating multipanel figures.

    :param axes: list of matplotlib Axis objects to enumerate
    :param size: Font size
    """

    for n, ax in enumerate(axes):

        ax.text(-0.15, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=size, weight='bold', usetex= False, family='sans-serif')


def get_stimvec(ISIvec, dt=0.1, null=0, extra=10):
    """
    Generates a binary stimulation vector from a vector with ISI intervals
    :param ISIvec: ISI interval vector (in ms)
    :param dt: timestep (ms)
    :param null: 0s in front of the vector (in ms)
    :param extra: 0s after the last stimulus (in ms)
    :return: binary stim vector
    """

    ISIindex = np.cumsum(np.round(np.array([i if i == 0 else i - dt for i in ISIvec]) / dt, 1))
    # ISI times accounting for base zero-indexing

    return np.array([0] * int(null / dt) + [1 if i in ISIindex.astype(int) else 0
                                            for i in np.arange(int(sum(ISIvec) / dt + extra / dt))]).astype(bool)


def get_ISIvec(freq, nstim):
    """
    Returns an ISI vector of a periodic stimulation train (constant frequency)
    :param freq: int of stimulation frequency
    :param nstim: number of stimuli in the train
    :return: ISI vector in ms
    """
    return [0] + list(np.array([1000 / freq]).astype(int)) * (nstim - 1)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print('Object pickled and saved.')


def load_pickle(filename):
    with open(filename, 'rb') as input:
        print('Here is your pickle. Enjoy.')
        return pickle.load(input)


def draw_windows(axis, start, end, color, alpha = 0.5):

    if start == end:
        axis.axvspan(start, end, color=color, alpha=1, lw=3)
    else:
        axis.axvspan(start, end, color=color, alpha=alpha, lw=0)

    return axis


def random_weights(n, min=0.0, max=1.0):
    return min + np.random.rand(n) * (max - min)


def poisson_simple(t, dt, r):
    """
    :param t: total number of timesteps
    :param dt: timestep in ms
    :param r: spiking rate in Hz
    :return: poisson spike train
    """
    draw = np.random.uniform(size=t)
    p = r * dt / 1000  # rate * timestep in seconds

    return (draw < p).astype(bool)  # returns binary poisson spike train

def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))
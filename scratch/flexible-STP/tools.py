import numpy as np
import pickle
import matplotlib.pyplot as plt
import string


def add_figure_letters(axes, size=14):
    """
    Function to add Letters enumerating multipanel figures.

    :param axes: list of matplotlib Axis objects to enumerate
    :param size: Font size
    """

    for n, ax in enumerate(axes):

        ax.text(
            -0.15,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=size,
            weight="bold",
            usetex=False,
            family="sans-serif",
        )


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


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def save_pickle(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print("Object pickled and saved.")


def load_pickle(filename):
    with open(filename, "rb") as input:
        print("Here is your pickle. Enjoy.")
        return pickle.load(input)


def draw_windows(axis, start, end, color, alpha=0.5):

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


def add_scalebar_ES(
    x_units=None,
    y_units=None,
    anchor=(0.98, 0.02),
    x_size=None,
    y_size=None,
    y_label_space=0.02,
    x_label_space=-0.02,
    bar_space=0.06,
    x_on_left=True,
    linewidth=3,
    remove_frame=True,
    omit_x=False,
    omit_y=False,
    round=True,
    usetex=True,
    ax=None,
):
    """
    Automagically add a set of x and y scalebars to a matplotlib plot
    Inputs:
        x_units: str or None
        y_units: str or None
        anchor: tuple of floats
        --  bottom right of the bbox (in axis coordinates)
        x_size: float or None
        --  Manually set size of x scalebar (or None for automatic sizing)
        y_size: float or None
        --  Manually set size of y scalebar (or None for automatic sizing)
        text_spacing: tuple of floats
        --  amount to offset labels from respective scalebars (in axis units)
        bar_space: float
        --  amount to separate bars from eachother (in axis units)
        linewidth: numeric
        --  thickness of the scalebars
        remove_frame: bool (default False)
        --  remove the bounding box, axis ticks, etc.
        omit_x/omit_y: bool (default False)
        --  skip drawing the x/y scalebar
        round: bool (default True)
        --  round units to the nearest integer
        ax: matplotlib.axes object
        --  manually specify the axes object to which the scalebar should be added
    """

    # Basic input processing.

    if ax is None:
        ax = plt.gca()

    if x_units is None:
        x_units = ""
    if y_units is None:
        y_units = ""

    # Do y scalebar.
    if not omit_y:

        if y_size is None:
            y_span = ax.get_yticks()[:2]
            y_length = y_span[1] - y_span[0]
            y_span_ax = ax.transLimits.transform(np.array([[0, 0], y_span]).T)[:, 1]
        else:
            y_length = y_size
            y_span_ax = ax.transLimits.transform(np.array([[0, 0], [0, y_size]]))[:, 1]
        y_length_ax = y_span_ax[1] - y_span_ax[0]

        if round:
            y_length = int(np.round(y_length))

        # y-scalebar label

        if y_label_space <= 0:
            horizontalalignment = "left"
        else:
            horizontalalignment = "right"

        if usetex:
            y_label_text = "${}${}".format(y_length, y_units)
        else:
            y_label_text = "{}{}".format(y_length, y_units)

        ax.text(
            anchor[0] - y_label_space,
            anchor[1] + y_length_ax / 2 + bar_space,
            y_label_text,
            verticalalignment="center",
            horizontalalignment=horizontalalignment,
            size="small",
            transform=ax.transAxes,
        )

        # y scalebar
        ax.plot(
            [anchor[0], anchor[0]],
            [anchor[1] + bar_space, anchor[1] + y_length_ax + bar_space],
            "k-",
            linewidth=linewidth,
            clip_on=False,
            transform=ax.transAxes,
        )

    # Do x scalebar.
    if not omit_x:

        if x_size is None:
            x_span = ax.get_xticks()[:2]
            x_length = x_span[1] - x_span[0]
            x_span_ax = ax.transLimits.transform(np.array([x_span, [0, 0]]).T)[:, 0]
        else:
            x_length = x_size
            x_span_ax = ax.transLimits.transform(np.array([[0, 0], [x_size, 0]]))[:, 0]
        x_length_ax = x_span_ax[1] - x_span_ax[0]

        if round:
            x_length = int(np.round(x_length))

        # x-scalebar label
        if x_label_space <= 0:
            verticalalignment = "top"
        else:
            verticalalignment = "bottom"

        if x_on_left:
            Xx_text_coord = anchor[0] - x_length_ax / 2 - bar_space
            Xx_bar_coords = [anchor[0] - x_length_ax - bar_space, anchor[0] - bar_space]
        else:
            Xx_text_coord = anchor[0] + x_length_ax / 2 + bar_space
            Xx_bar_coords = [anchor[0] + x_length_ax + bar_space, anchor[0] + bar_space]

        if usetex:
            x_label_text = "${}${}".format(x_length, x_units)
        else:
            x_label_text = "{}{}".format(x_length, x_units)

        ax.text(
            Xx_text_coord,
            anchor[1] + x_label_space,
            x_label_text,
            verticalalignment=verticalalignment,
            horizontalalignment="center",
            size="small",
            transform=ax.transAxes,
        )

        # x scalebar
        ax.plot(
            Xx_bar_coords,
            [anchor[1], anchor[1]],
            "k-",
            linewidth=linewidth,
            clip_on=False,
            transform=ax.transAxes,
        )

    if remove_frame:
        ax.axis("off")

import numpy as np
import pickle
import string

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# PLOT SETTINGS
# # # # # # # # # #
plt.style.use('spiffy')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc("font", size=7, family='serif')
plt.rc('text', usetex=True)

markersize = 3
capsize = 2
lw = 1
lw_traces = 0.5
figsize = (5.25102, 5.25102 * 1 / 3)  # From LaTeX readout of textwidth

# COLORSCHEME: RED/BLUE
c_12ca = "#cc3311"
c_12ca_trace = "#f9c7bb"
c_25ca = "#0077bb"
c_25ca_trace = "#bbe6ff"


# FUNCTIONS
# # # # # # # # # #


def add_scalebar(
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


def load_pickle(filename):
    with open(filename, "rb") as input:
        print("Here is your pickle. Enjoy.")
        return pickle.load(input)


def plot():
    # Make Figure Grid
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=10, nrows=1, figure=fig, wspace=0.1, hspace=0.1)
    axLeft = fig.add_subplot(spec[0, 0:3])
    subspec1 = spec[0, 1:4].subgridspec(ncols=1, nrows=2, wspace=0.0, hspace=0.0)
    axTrace1 = fig.add_subplot(subspec1[0, 0])
    axTrace2 = fig.add_subplot(subspec1[1, 0])
    axAmp = fig.add_subplot(spec[0, 4:7])
    axCV = fig.add_subplot(spec[0, 7:])

    axLeft.axis("off")
    plot_traces(axTrace1, axTrace2)
    axAmp = plot_amps(axAmp)
    axCV = plot_cv(axCV)

    add_figure_letters([axLeft, axAmp, axCV], size=10)

    return fig


def plot_traces(ax1, ax2):

    xax = np.arange(0, traces["1.2ca"].mean(0).shape[0] * 0.1, 0.1)
    for trace in traces["1.2ca"]:
        ax1.plot(xax[::2], trace[::2], color=c_12ca_trace, lw=lw_traces)
    ax1.plot(xax[::2], traces["1.2ca"].mean(0)[::2], color=c_12ca, lw=lw_traces)
    # ax1.set_xticks([])
    ax1.set_ylim(bottom=np.min(traces["1.2ca"]), top=50)
    ax1.set_title("1.2 mM $[Ca^{2+}]$", color=c_12ca, fontdict={"fontsize": 8})
    add_scalebar(
        x_units="ms",
        y_units="pA",
        anchor=(0.3, 0.1),
        # x_size=20,
        y_size=100,
        y_label_space=0.02,
        x_label_space=-0.1,
        bar_space=0,
        x_on_left=False,
        linewidth=0.5,
        remove_frame=True,
        omit_x=False,
        omit_y=False,
        round=True,
        usetex=True,
        ax=ax1,
    )

    for trace in traces["2.5ca"]:
        ax2.plot(xax[::2], trace[::2], color=c_25ca_trace, lw=lw_traces)
    ax2.plot(xax[::2], traces["2.5ca"].mean(0)[::2], color=c_25ca, lw=lw_traces)
    # ax2.set_xticks([])
    ax2.set_ylim(bottom=np.min(traces["2.5ca"]), top=50)
    ax2.set_title("2.5 mM $[Ca^{2+}]$", color=c_25ca, fontdict={"fontsize": 8})
    add_scalebar(
        x_units="ms",
        y_units="pA",
        anchor=(0.3, 0.15),
        # x_size=20,
        y_size=500,
        y_label_space=0.02,
        x_label_space=-0.1,
        bar_space=0,
        x_on_left=False,
        linewidth=0.5,
        remove_frame=True,
        omit_x=False,
        omit_y=False,
        round=True,
        usetex=True,
        ax=ax2,
    )

    return ax1, ax2


def plot_amps(ax):

    x_ax = np.arange(1, 6)
    ax.errorbar(
        x_ax,
        amps["1.2ca"],
        amps["1.2ca_error"],
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_12ca,
    )
    ax.errorbar(
        x_ax,
        amps["2.5ca"],
        amps["2.5ca_error"],
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_25ca,
    )
    ax.set_ylim(bottom=0, top=1000)
    ax.set_yticks([0, 500, 1000])
    ax.set_xticks(x_ax)
    ax.set_ylabel("EPSC amplitude (pA)")
    ax.set_xlabel(r"stimulus nr.")

    return ax


def plot_cv(ax):

    x_ax = np.arange(1, 6)
    ax.errorbar(
        x_ax,
        cvs["1.2ca"],
        cvs["1.2ca_error"],
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_12ca,
    )
    ax.errorbar(
        x_ax,
        cvs["2.5ca"],
        cvs["2.5ca_error"],
        capsize=capsize,
        marker="s",
        lw=lw,
        elinewidth=lw * 0.7,
        markersize=markersize,
        color=c_25ca,
    )
    ax.set_ylim(bottom=0.2, top=0.8)
    ax.set_xticks(x_ax)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_ylabel("CV")
    ax.set_xlabel(r"stimulus nr.")

    return ax


if __name__ == "__main__":
    import os, inspect

    # FIGURE 1 PLOTTING

    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)

    # DATA IMPORT
    # # # # # # # # # #
    amps = load_pickle(parent_dir + "/data/processed/chamberland2014/fig2_amps.pkl")
    cvs = load_pickle(parent_dir + "/data/processed/chamberland2014/fig2_cvs.pkl")
    traces = load_pickle(parent_dir + "/data/processed/chamberland2014/fig2_traces.pkl")

    fig = plot()
    fig.savefig(current_dir + "/figures/Fig2_raw.pdf", bbox_inches='tight')
    plt.show()

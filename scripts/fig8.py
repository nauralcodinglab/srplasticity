# General
import pickle
from pathlib import Path
import os, inspect
import numpy as np
import string

# Models
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from srplasticity.srp import ExpSRP, ExponentialKernel
from srplasticity.inference import fit_srp_model, fit_srp_model_gridsearch

# Plotting
from spiffyplots import MultiPanel
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("spiffy")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Set to True to fit model parameters
# Set to False to load fitted parameters from `scripts / modelfits`
fitting_tm = False
fitting_srp = False

# Total test sets (with independent fits each)
test_keys = ["invivo", "100", "20", "20100", "111", "10100", "10020"]

# Paths
current_dir = Path(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)
parent_dir = Path(os.path.dirname(current_dir))

modelfit_dir = current_dir / "modelfits"
data_dir = parent_dir / "data" / "processed" / "chamberland2018"
figure_dir = current_dir / "figures"
supplement_dir = current_dir / "supplements"

# Plots
plotted_testsets_ordered = ["20", "20100", "invivo"]

matplotlib.rc("xtick", top=False)
matplotlib.rc("ytick", right=False)
matplotlib.rc("ytick.minor", visible=False)
matplotlib.rc("xtick.minor", visible=False)
plt.rc("font", size=8)
plt.rc("text", usetex=True)

figsize = (5.25102 * 1.5, 5.25102 * 1.5)  # From LaTeX readout of textwidth

color = {"tm": "blue", "srp": "darkred"}
c_kernels = ("#525252", "#969696", "#cccccc")  # greyscale

markersize = 2
capsize = 2
lw = 1

protocol_names = {
    "100": "10 x 100 Hz",
    "20": "10 x 20 Hz",
    "111": "6 x 111 Hz",
    "20100": "5 x 20 Hz + 1 x 100 Hz",
    "10100": "5 x 10 Hz + 1 x 100 Hz",
    "10020": "5 x 100 Hz + 1 x 20 Hz",
    "invivo": "in-vivo burst",
}

short_names = {
    "100": "100 Hz",
    "20": "20 Hz",
    "111": "111 Hz",
    "20100": "20/100 Hz",
    "10100": "10/100 Hz",
    "10020": "100/20 Hz",
    "invivo": "in-vivo",
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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
    with open(filename, "rb") as file:
        print("Here is your pickle. Enjoy.")
        return pickle.load(file)


def save_pickle(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print("Object pickled and saved.")


def get_model_estimates(model, stimulus_dict):
    """
    :return: Model estimates for training dataset
    """
    estimates = {}
    if isinstance(model, ExpSRP):
        means = {}
        sigmas = {}
        for key, isivec in stimulus_dict.items():
            means[key], sigmas[key], estimates[key] = model.run_ISIvec(
                isivec, ntrials=10000
            )
        return means, sigmas, estimates

    elif isinstance(model, TsodyksMarkramModel):
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)
            model.reset()

        return estimates

    else:
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)

        return estimates


def mse(targets, estimate):
    """
    :param targets: 2D np.array with response amplitudes of shape [n_sweep, n_stimulus]
    :param estimate: 1D np.array with estimated response amplitudes of shape [n_stimulus]
    :return: mean squared errors
    """
    return np.nansum((targets - estimate) ** 2) / np.count_nonzero(~np.isnan(targets))


def mse_by_protocol(target_dict, estimates_dict):
    """
    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: mse by protocol
    """
    loss = {}
    for key in target_dict.keys():
        loss[key] = mse(target_dict[key], estimates_dict[key])

    return loss


def sterr(mat):
    """
    standard error of the mean
    :param mat: A matrix of [n_samples, n_spikes]
    """
    return np.nanstd(mat, 0) / np.sqrt(np.count_nonzero(~np.isnan(mat), 0))


def get_train_dict(targets, test_key):
    """
    Get training dictionary based on the test key
    """

    return {key: targets[key] for key in protocol_names.keys() if key != test_key}


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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# LOADING DATA FROM CHAMBERLAND ET AL. (2018)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ISI vectors
stimulus_dict = {
    "20": [0] + [50] * 9,
    "100": [0] + [10] * 9,
    "20100": [0, 50, 50, 50, 50, 10],
    "10020": [0, 10, 10, 10, 10, 50],
    "10100": [0, 100, 100, 100, 100, 10],
    "111": [0] + [5] * 5,
    "invivo": [0, 6, 90.9, 12.5, 25.6, 9],
}

# Response amplitudes
target_dict = {}
for key in stimulus_dict:
    target_dict[key] = load_pickle(
        Path(data_dir / str(key + "_normalized_by_cell.pkl"))
    )
    # set zero values to nan
    target_dict[key][target_dict[key] == 0] = np.nan

# Noise correlation data
noisecor_data = {}
for key in stimulus_dict:
    noisecor_data[key] = load_pickle(
        Path(data_dir / "noisecorrelations" / str(key + ".pkl"))
    )

# example trace
example_trace = load_pickle(
        Path(data_dir / 'example_trace.pkl')
    )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING TM MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if fitting_tm:
    print("Fitting TM model to Chamberland et al. (2018) data...")

    # Tsodyks-Markram model parameter ranges
    tm_param_ranges = (
        slice(0.001, 0.0105, 0.0005),  # U
        slice(0.001, 0.0105, 0.0005),  # f
        slice(1, 501, 10),  # tau_u
        slice(1, 501, 10),  # tau_r
    )

    # Step 1: Fitting to whole dataset
    print("Fitting TM model to all protocols...")
    tm_params, tm_sse, grid, sse_grid = fit_tm_model(
        stimulus_dict,
        target_dict,
        tm_param_ranges,
        disp=True,  # display output
        workers=-1,  # split over all available CPU cores
        full_output=True,  # save function value at each grid node
    )

    # Save fitted TM model parameters
    save_pickle(tm_params, modelfit_dir / "chamberland2018_TMmodel.pkl")

    # Step 2: Holding out test sets defined in test_keys one at a time
    tm_testparams = {}
    for testkey in test_keys:
        print("Holding out {} Hz data".format(testkey))

        tm_testparams[testkey], _, _, _ = fit_tm_model(
            stimulus_dict,
            get_train_dict(target_dict, testkey),
            tm_param_ranges,
            disp=True,  # display output
            workers=-1,  # split over all available CPU cores
            full_output=True,  # save function value at each grid node
        )
        print("FITTED PARAMETERS:")
        print(tm_testparams[testkey])

    save_pickle(tm_testparams, modelfit_dir / "chamberland2018_TMmodel_validation.pkl")

else:
    print("Loading fitted TM model parameters...")
    tm_params = load_pickle(modelfit_dir / "chamberland2018_TMmodel.pkl")
    tm_testparams = load_pickle(modelfit_dir / "chamberland2018_TMmodel_validation.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING SRP MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if fitting_srp:
    print("Fitting SRP model to Chamberland et al. (2018) data...")

    # Pre-define fixed parameters
    mu_kernel_taus = [15, 100, 650]
    sigma_kernel_taus = [15, 100, 650]

    # Initial guess for sigma scale
    sigma_scale = 4

    # Parameter ranges for grid search. Total of 128 initial starts
    srp_param_ranges = (
        slice(-3, 1, 0.25),  # both baselines
        slice(-2, 2, 0.25),  # all amplitudes (weighted by tau in fitting procedure)
    )

    # Step 1: Fitting to whole dataset
    print("Fitting SRP model to all protocols...")
    srp_params, bestfit, _starts, _fvals, _allsols = fit_srp_model_gridsearch(
        stimulus_dict,
        target_dict,
        mu_kernel_taus,
        sigma_kernel_taus,
        param_ranges=srp_param_ranges,
        mu_scale=None,
        sigma_scale=4,
        bounds="default",
        method="L-BFGS-B",
        workers=-1,
        options={"maxiter": 500, "disp": False, "ftol": 1e-12, "gtol": 1e-9},
    )
    print("BEST SOLUTION:")
    print(bestfit)

    # Save fitted SRP model parameters
    save_pickle(srp_params, modelfit_dir / "chamberland2018_SRPmodel.pkl")

    # Step 2: Holding out test sets defined in test_keys one at a time
    srp_testparams = {}
    for testkey in test_keys:
        print("Holding out {} Hz data".format(testkey))

        srp_testparams[testkey], _bestfit, _, _, _ = fit_srp_model_gridsearch(
            stimulus_dict,
            get_train_dict(target_dict, testkey),
            mu_kernel_taus,
            sigma_kernel_taus,
            param_ranges=srp_param_ranges,
            mu_scale=None,
            sigma_scale=4,
            bounds="default",
            method="L-BFGS-B",
            workers=-1,
            options={"maxiter": 500, "disp": False, "ftol": 1e-12, "gtol": 1e-9},
        )
        print("BEST SOLUTION:")
        print(_bestfit)

    save_pickle(
        srp_testparams, modelfit_dir / "chamberland2018_SRPmodel_validation.pkl"
    )

else:
    print("Loading fitted SRP model parameters...")
    srp_params = load_pickle(modelfit_dir / "chamberland2018_SRPmodel.pkl")
    srp_testparams = load_pickle(
        modelfit_dir / "chamberland2018_SRPmodel_validation.pkl"
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# COMPUTE MODEL ESTIMATES AND PERFORMANCE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

tm_est = get_model_estimates(TsodyksMarkramModel(*tm_params), stimulus_dict)
srp_mean, srp_sigma, srp_est = get_model_estimates(ExpSRP(*srp_params), stimulus_dict)

tm_test = {"est": {}, "mse": {}, "msenorm": {}, "testmse": 0}
srp_test = {"mean": {}, "sigma": {}, "est": {}, "mse": {}, "msenorm": {}, "testmse": 0}

datameans = {key: np.nanmean(x, 0) for key, x in target_dict.items()}
data_mse = mse_by_protocol(target_dict, datameans)

for testkey in test_keys:

    # All estimates
    tm_test["est"][testkey] = get_model_estimates(
        TsodyksMarkramModel(*tm_testparams[testkey]), stimulus_dict
    )
    tm_test["mse"][testkey] = mse_by_protocol(target_dict, tm_test["est"][testkey])[
        testkey
    ]

    (
        srp_test["mean"][testkey],
        srp_test["sigma"][testkey],
        srp_test["est"][testkey],
    ) = get_model_estimates(ExpSRP(*srp_testparams[testkey]), stimulus_dict)
    srp_test["mse"][testkey] = mse_by_protocol(target_dict, srp_test["mean"][testkey])[
        testkey
    ]

    tm_test["msenorm"][testkey] = tm_test["mse"][testkey] / data_mse[testkey]
    srp_test["msenorm"][testkey] = srp_test["mse"][testkey] / data_mse[testkey]

    # Sum up test set errors
    tm_test["testmse"] += tm_test["msenorm"][testkey]
    srp_test["testmse"] += srp_test["msenorm"][testkey]

# Stuff for plotting
srp_kernels = ExponentialKernel(srp_params[2], srp_params[1])._all_exponentials
srp_sigma_kernels = ExponentialKernel(srp_params[5], srp_params[4])._all_exponentials


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def plot_allNoiseCorrelations():
    """
    Investigate all noise correlations
    """

    fig = MultiPanel(
        grid=[len(noisecor_data.keys())], figsize=(3 * len(noisecor_data.keys()), 3)
    )
    panel = 0
    for key in noisecor_data.keys():
        paired = noisecor_data[key]

        fig.panels[panel].hlines(y=0, xmax=4, xmin=-4, color="darkred")
        fig.panels[panel].vlines(x=0, ymax=4, ymin=-4, color="darkred")
        fig.panels[panel].scatter(paired[:, 0], paired[:, 1], s=2, c="black")
        fig.panels[panel].set_xlabel(r"$\Delta x(s)$ (in SD)")
        fig.panels[panel].set_ylabel(r"$\Delta x(s+1)$ (in SD)")
        fig.panels[panel].set_ylim(-4, 4)
        fig.panels[panel].set_xlim(-4, 4)
        fig.panels[panel].set_xticks([-4, -2, 0, 2, 4])
        fig.panels[panel].set_yticks([-4, -2, 0, 2, 4])
        fig.panels[panel].set_title(protocol_names[key])
        panel += 1

    plt.tight_layout()
    plt.savefig(supplement_dir / "Supp_Fig8_noisecorrelation_by_protocol.pdf")
    plt.show()


def plot_allfits():

    npanels = len(list(target_dict.keys()))
    fig = MultiPanel(grid=[npanels, npanels], figsize=(npanels * 3, 6))

    # Plot mean fit
    for ix, key in enumerate(list(target_dict.keys())):
        xax = np.arange(1, len(tm_est[key]) + 1)
        standard_error = sterr(target_dict[key])
        fig.panels[ix].errorbar(
            xax,
            np.nanmean(target_dict[key], 0),
            yerr=standard_error,
            color="black",
            marker="o",
            markersize=2,
            label="Data",
        )
        fig.panels[ix].plot(xax, tm_est[key], color=color["tm"], label="TM model")
        fig.panels[ix].plot(xax, srp_mean[key], color=color["srp"], label="SRP model")
        fig.panels[ix].set_title(protocol_names[key])
        fig.panels[ix].set_xticks(xax)
        fig.panels[ix].set_ylim(0.5, 9)
        fig.panels[ix].set_yticks([1, 3, 5, 7, 9])

    # Plot sigma fit

    for ix2, key in enumerate(list(target_dict.keys())):
        xax = np.arange(1, len(tm_est[key]) + 1)
        fig.panels[ix + 1 + ix2].plot(
            xax,
            np.nanstd(target_dict[key], 0),
            color="black",
            marker="o",
            markersize=2,
            label="data",
        )
        fig.panels[ix + 1 + ix2].plot(
            xax, srp_sigma[key], color=color["srp"], label="SRP model"
        )
        fig.panels[ix + 1 + ix2].set_xticks(xax)
        fig.panels[ix + 1 + ix2].set_xlabel("spike nr")
        fig.panels[ix + 1 + ix2].set_ylim(0, 8)
        fig.panels[ix + 1 + ix2].set_yticks([0, 2, 4, 6, 8])

    fig.panels[ix + 1].legend(frameon=False)
    fig.panels[ix + 1].set_ylabel(r"std. $\sigma$")
    fig.panels[0].legend(frameon=False)
    fig.panels[0].set_ylabel("norm. EPSC amplitude")

    plt.savefig(supplement_dir / "Supp_Fig8_all_model_fits.pdf")
    plt.show()


def plot_kernel(axis):
    t_max = 1000  # in ms
    t_inset = 1000
    x_ax = np.arange(0, t_max, 0.1)
    inset_x_ax = np.arange(0, t_inset, 0.1)

    inset_ax = axis.inset_axes([0.5, 0.5, 0.45, 0.45])

    flipped = np.flip(srp_kernels, 0)
    flipped_sigma = np.flip(srp_sigma_kernels, 0)
    for ix, kernel in enumerate(srp_kernels):
        label = str(srp_params[2][2-ix])
        axis.fill_between(
            x_ax,
            0,
            np.cumsum(flipped[:, : t_max * 10], 0)[ix],
            facecolor=c_kernels[2 - ix],
            alpha=1,
            zorder=3 - ix,
            label=label,
        )
        inset_ax.fill_between(
            inset_x_ax,
            0,
            np.cumsum(flipped_sigma[:, : t_inset * 10], 0)[ix],
            facecolor=c_kernels[2 - ix],
            alpha=1,
            zorder=3 - ix,
        )

    sum = np.sum(srp_kernels, 0)
    sum_sigma = np.sum(srp_sigma_kernels, 0)
    axis.plot(x_ax, sum[: t_max * 10], c="black", lw=lw)
    inset_ax.plot(inset_x_ax, sum_sigma[: t_inset * 10], c="black", lw=lw)

    # Inset with first 100 ms
    axis.set_xlabel("t (ms)")
    axis.set_xticks([0, t_max])
    axis.set_yticks([0, 1])
    inset_ax.set_xlabel("t(ms)")
    inset_ax.set_xticks([0, t_inset])
    inset_ax.set_yticks([0, 1])

    axis.legend(frameon=False)

    # axis.axis('off')


def plot_comparative_fits(axes):
    """
    :param axes: list of 3 axes
    """

    for ax in axes:
        ax.set_ylim(0.5, 9)
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.set_xlabel("spike nr.")

    axes[0].set_ylabel("norm. EPSC")

    for ix, testkey in enumerate(plotted_testsets_ordered):
        # xax = np.arange(1, len(tm_est[testkey]) + 1)
        xax = np.arange(1, 7)
        standard_error = sterr(target_dict[testkey])
        axes[ix].errorbar(
            xax,
            np.nanmean(target_dict[testkey][:, :6], 0),
            yerr=standard_error[:6],
            color="black",
            label="Data",
            capsize=capsize,
            marker="s",
            lw=lw,
            elinewidth=lw * 0.7,
            markersize=markersize,
        )

        axes[ix].plot(
            xax,
            tm_test["est"][testkey][testkey][:6],
            color=color["tm"],
            label="TM model",
        )
        axes[ix].plot(
            xax,
            srp_test["mean"][testkey][testkey][:6],
            color=color["srp"],
            label="SRP model",
        )
        axes[ix].set_xticks(xax[::2])
        # axes[ix].set_title(protocol_names[testkey])

        if ix > 0:
            axes[ix].set_yticklabels([])
    axes[0].legend(frameon=False)


def plot_protocols_data(axis):

    datamean = np.nanmean(target_dict['100'],0)[:6]
    axis.plot(target_dict['100'].mean())


def plot_mse(axis):

    # Make data for barplot
    labels = np.array(["TM model", "SRP model"])
    height = np.array([tm_test["testmse"] / 7, srp_test["testmse"] / 7])

    x = np.arange(2)  # the label locations

    axis.bar(x[0], height[0], color=color["tm"])
    axis.bar(x[1], height[1], color=color["srp"])

    axis.set_ylim(1, 1.1)
    axis.set_yticks([1, 1.1])
    axis.set_ylabel("test MSE (norm.)")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)


def plot_noisecor(axis):
    stackdata = np.vstack([pair for pair in noisecor_data.values()])

    axis.hlines(y=0, xmax=4, xmin=-4, color="darkred")
    axis.vlines(x=0, ymax=4, ymin=-4, color="darkred")
    axis.scatter(stackdata[:, 0], stackdata[:, 1], s=0.5, c="black")
    axis.set_xlabel(r"$S_j$ amplitude deviation (std)")
    axis.set_ylabel(r"$S_{j+1}$ amplitude deviation (std)")
    axis.set_ylim(-4, 4)
    axis.set_xlim(-4, 4)
    axis.set_xticks([-4, -2, 0, 2, 4])
    axis.set_yticks([-4, -2, 0, 2, 4])


def plot_std(axis):

    axis.plot(np.nanstd(target_dict['20'], 0), color='black', ls='dashed', marker='o', label = '20 Hz')
    axis.plot(np.nanstd(target_dict['100'], 0), color='black', marker='s', label ='100 Hz')

    axis.plot(srp_sigma['20'], color=color['srp'], ls='dashed')
    axis.plot(srp_sigma['100'], color=color['srp'])

    axis.set_ylabel(r'sdt. deviation $\sigma$')
    axis.set_xlabel('nr. spike')

    axis.legend(frameon=False)


def plot_traces(axis):

    x_ax = np.arange(0, len(example_trace) * 0.1 , 0.1)
    axis.plot(x_ax, example_trace, lw=0.5, color='lightgrey')
    axis.plot(x_ax, example_trace.mean(1), lw=0.5, color='black')
    axis.set_ylim(-2000, 2500)
    add_scalebar(
        x_units="ms",
        y_units="pA",
        anchor=(0.05, 0.05),
        x_size=None,
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
        ax=axis)


def plot_fig8():
    """
    Plots figure 8
    """
    axes_with_letter = [0, 1, 2, 3, 4, 5, 6, 9]
    fig = MultiPanel(grid=[3, 3, 4], figsize=figsize)

    plot_traces(fig.panels[0])
    plot_noisecor(fig.panels[2])

    # SRP Model fit
    plot_kernel(fig.panels[3])
    # plot_traces(model, fig.panels[4)
    plot_std(fig.panels[5])

    # SRP / TM model fits
    plot_comparative_fits([fig.panels[ix] for ix in np.arange(6, 9)])

    # MSE
    plot_mse(fig.panels[9])
    # plot_totalmse(fig.panels[10)

    add_figure_letters([fig.panels[ix] for ix in axes_with_letter], 12)
    plt.tight_layout()

    plt.savefig(figure_dir / "Fig8.pdf")
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING SCRIPT
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    from srplasticity.srp import DetSRP, ExponentialKernel

    # Plot of fits to all protocols when both models were fit to the whole dataset
    #plot_allfits()
    #plot_allNoiseCorrelations()
    plot_fig8()

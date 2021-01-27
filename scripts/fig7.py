# General
from pathlib import Path
import os
import inspect
import pickle

# numpy / scipy
import numpy as np
import scipy.stats

# Models
from srplasticity.srp import ExpSRP, ExponentialKernel, _convolve_spiketrain_with_kernel
from srplasticity.inference import fit_srp_model, _nll
from srplasticity.tools import get_stimvec

# Plotting
from spiffyplots import MultiPanel
import matplotlib.pyplot as plt
from matplotlib import gridspec

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

COMPUTE_NLL_LANDSCAPE = False  # Set to False to load from disk, or to true to recompute the NLL landscape (C-F)
COMPUTE_ALL_FITS = False  # Set to False to load from disk, or to true to fit models to different sized spiketrains

# Seed for example spiketrain
np.random.seed(2021)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS FOR EXAMPLE SPIKETRAIN
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Spike train
rate = 10  # firing rate of poisson input (Hz)
nspikes = 200  # number of input spikes
ntrials = 20
dt = 0.1

# Simulated PSCs
PSCtau = 50
PSCamp = -10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# TRUE MODEL PARAMETERS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Model
mu_tau = 100  # Mu kernel time constant
mu_amp = 300  # Mu amplitude
mu_baseline = -2  # sigma baseline
sigma_tau = 100  # Sigma kernel time constant
sigma_amp = 200  # sigma amplitude
sigma_baseline = -2  # sigma baseline
sigma_scale = 10  # sigma scale

true_parameters = {
    "mu_baseline": mu_baseline,
    "mu_amps": mu_amp,
    "mu_taus": mu_tau,
    "sigma_baseline": sigma_baseline,
    "sigma_amps": sigma_amp,
    "sigma_taus": sigma_tau,
    "mu_scale": None,
    "sigma_scale": sigma_scale,
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS FOR INFERENCE AND GRID SEARCH
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

n_gridnodes = 100
initial_guess = [-2, 300, -2, 200, 10]

training_nspikes = np.logspace(1, 3, 8).astype(int)
training_nsweeps = 1
training_niterations = 20

testing_nspikes = 100
testing_nsweeps = 20

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PATHS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

current_dir = Path(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)
parent_dir = Path(os.path.dirname(current_dir))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

plot_tmax = 15000

color = {
    "inferred": "#cc3311",
    "first": "#EE7733",
    "second": "#0077BB",
    "third": "#009988",
}

plt.style.use("spiffy")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rc("font", size=7, family="serif")
plt.rc("text", usetex=True)

figsize = (5.25102, 5.25102 * 1.05)  # From LaTeX readout of textwidth

markersize = 3
capsize = 2
lw = 1

nll_landscapes = {
    "C": {"variable_parameters": ("sigma_amps", "mu_amps")},
    "D": {"variable_parameters": ("sigma_scale", "sigma_baseline")},
    "E": {"variable_parameters": ("sigma_baseline", "mu_baseline")},
    "F": {"variable_parameters": ("sigma_scale", "sigma_amps")},
}

# parameter names for axes labels
parameter_names = {
    "mu_baseline": r"$\mu$ baseline",
    "mu_amps": r"$\mu$ amplitude",
    "sigma_baseline": r"$\sigma$ baseline",
    "sigma_amps": r"$\sigma$ amplitude",
    "sigma_scale": r"$\sigma$ scale",
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_pickle(filename):
    with open(filename, "rb") as file:
        print("Here is your pickle. Enjoy.")
        return pickle.load(file)


def save_pickle(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print("Object pickled and saved.")


def get_spiketrain(T, dt, rate):
    """
    poisson spike train
    :param T: Time in milliseconds
    :param dt: timestep in milliseconds
    :rate: firing rate in Hz
    """
    spikeprob_per_timestep = rate * dt / 1000
    spikes = np.random.rand(int(T / dt)) < spikeprob_per_timestep

    return spikes


def get_poisson_ISIs(nspikes, rate):
    """
    poisson ISIs
    :param nspikes: number of spikes
    :param rate: firing rate in Hz
    """

    meanISI = int(1000 / rate)
    isis = np.random.exponential(scale=meanISI, size=nspikes).round(1)
    isis[isis < 2] = 2  # minimum 2ms refractory period
    return isis


def fitSRPmodel(stimulation, target, **kwargs):

    inferred_parameters, fitting_res = fit_srp_model(
        initial_guess,
        {"surrogate": stimulation},
        {"surrogate": target},
        mu_taus=[mu_tau],
        sigma_taus=[sigma_tau],
        mu_scale=None,
        bounds="default",
        loss="equal",
        options={"maxiter": 500, "disp": False, "ftol": 1e-12, "gtol": 1e-9},
        **kwargs
    )

    return inferred_parameters, fitting_res


def nll_gridsearch(variable_params):

    assert len(variable_params) == 2

    # Check landscape around true value
    xTrue, yTrue = [true_parameters[parameter] for parameter in variable_params]
    x = np.linspace(xTrue * 0.5, xTrue * 1.5, n_gridnodes)
    y = np.linspace(yTrue * 0.5, yTrue * 1.5, n_gridnodes)
    xgrid, ygrid = np.meshgrid(x, y)
    nll = np.zeros((n_gridnodes, n_gridnodes))

    for xix, xvalue in enumerate(x):
        for yix, yvalue in enumerate(y):

            # make model with parameters
            temp = true_parameters.copy()
            temp[variable_params[0]] = xvalue
            temp[variable_params[1]] = yvalue
            model = ExpSRP(**temp)

            means, sigmas, _ = model.run_ISIvec(ISIs)
            nll[xix, yix] = _nll(efficacies_true, means, sigmas)

    return {"xgrid": xgrid, "ygrid": ygrid, "nll": nll}


def mse(targets, estimate):
    """
    :param targets: 2D np.array with response amplitudes of shape [n_sweep, n_stimulus]
    :param estimate: 1D np.array with estimated response amplitudes of shape [n_stimulus]
    :return: mean squared errors
    """
    return np.nansum((targets - estimate) ** 2) / np.count_nonzero(~np.isnan(targets))


def get_training_targets():
    trainingsets = {}
    for n in training_nspikes:
        trainingsets[n] = list()
        for _ in range(training_niterations):

            # get isi vector
            isivec = list(get_poisson_ISIs(n, rate))

            # compute efficacies
            _, _, eff = model_true.run_ISIvec(isivec, training_nsweeps)

            trainingsets[n].append({"targets": eff, "stim": isivec})

    return trainingsets


def calculate_parameter_error(trainingfits):

    errors = {
        "mu_baseline": np.zeros((len(training_nspikes), training_niterations)).T,
        "mu_amps": np.zeros((len(training_nspikes), training_niterations)).T,
        "sigma_baseline": np.zeros((len(training_nspikes), training_niterations)).T,
        "sigma_amps": np.zeros((len(training_nspikes), training_niterations)).T,
        "sigma_scale": np.zeros((len(training_nspikes), training_niterations)).T,
    }
    for ix, n in enumerate(training_nspikes):
        paramlist = trainingfits[n]["params"]

        errors["mu_baseline"][:, ix] = np.abs(
            (np.array([i[0] for i in paramlist]) - mu_baseline) / mu_baseline
        )
        errors["mu_amps"][:, ix] = np.abs(
            (np.array([i[1][0] for i in paramlist]) - mu_amp) / mu_amp
        )
        errors["sigma_baseline"][:, ix] = np.abs(
            (np.array([i[3] for i in paramlist]) - sigma_baseline) / sigma_baseline
        )
        errors["sigma_amps"][:, ix] = np.abs(
            (np.array([i[4][0] for i in paramlist]) - sigma_amp) / sigma_amp
        )
        errors["sigma_scale"][:, ix] = np.abs(
            (np.array([i[-1] for i in paramlist]) - sigma_scale) / sigma_scale
        )

    return errors


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# GENERATE SURROGATE DATA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ISIs = list(get_poisson_ISIs(nspikes, rate))
spiketrain = get_stimvec(ISIs, dt, 0, 0)

# True model from defined parameters
model_true = ExpSRP(
    mu_baseline,
    [mu_amp],
    [mu_tau],
    sigma_baseline,
    [sigma_amp],
    [sigma_tau],
    None,
    sigma_scale,
)
mu_true, sigma_true, efficacies_true, efficacytrains_true = model_true.run_spiketrain(
    spiketrain, ntrials=ntrials
)
data_nll = _nll(efficacies_true, mu_true, sigma_true)

# simulated PSCs
PSCkernel = ExponentialKernel(PSCtau, PSCamp).kernel
PSCs = np.vstack(
    [_convolve_spiketrain_with_kernel(x, PSCkernel) for x in efficacytrains_true]
)
PSCs += 0.05 * np.random.randn(PSCs.size).reshape(PSCs.shape)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# NLL LANDSCAPE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if COMPUTE_NLL_LANDSCAPE:
    for letter in nll_landscapes.keys():
        nll_landscapes[letter].update(
            nll_gridsearch(nll_landscapes[letter]["variable_parameters"])
        )

        # Save pickle
        save_pickle(
            nll_landscapes[letter],
            current_dir / "nll_landscape" / "panel_{}.pickle".format(letter),
        )
else:
    for letter in nll_landscapes.keys():

        # Load pickle
        loaded = load_pickle(
            current_dir / "nll_landscape" / "panel_{}.pickle".format(letter)
        )
        nll_landscapes[letter].update(loaded)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETER INFERENCE: EXAMPLE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

inferred_parameters, fitting_res = fitSRPmodel(ISIs, efficacies_true)
model_inferred = ExpSRP(*inferred_parameters)

(
    mu_inferred,
    sigma_inferred,
    efficacies_inferred,
    efficacytrains_inferred,
) = model_inferred.run_spiketrain(spiketrain, ntrials=ntrials)

# Inferred PSCs
PSCs_inferred = np.vstack(
    [_convolve_spiketrain_with_kernel(x, PSCkernel) for x in efficacytrains_inferred]
)
PSCs_inferred += 0.05 * np.random.randn(PSCs.size).reshape(PSCs.shape)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETER INFERENCE: QUANTIFICATION
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
trainingsets = get_training_targets()

teststim = list(get_poisson_ISIs(testing_nspikes, rate))
_, _, testtargets = model_true.run_ISIvec(teststim, testing_nsweeps)

if COMPUTE_ALL_FITS:
    trainingfits = {}
    for key in training_nspikes:
        params = []
        testmse = []
        for ix in range(training_niterations):

            # parameter inference on training set
            inferred_parameters, _ = fitSRPmodel(
                trainingsets[key][ix]["stim"], trainingsets[key][ix]["targets"]
            )

            test_estimates, _, _ = ExpSRP(*inferred_parameters).run_ISIvec(teststim)

            testmse.append(mse(testtargets, test_estimates))
            params.append(inferred_parameters)

        trainingfits[key] = {"params": params, "mse": np.array(testmse)}

    save_pickle(trainingfits, current_dir / "modelfits" / "fig7_allfits.pkl")

else:
    trainingfits = load_pickle(current_dir / "modelfits" / "fig7_allfits.pkl")

parameter_errors = calculate_parameter_error(trainingfits)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot_spiketrain(axis, spktr):
    axis.plot(spktr, lw=lw, color="black")
    axis.axis("off")


def plot_PSCs(axis1, axis2):
    axis1.plot(
        PSCs.T[:plot_tmax][::5], label="data", color="black", lw=lw * 0.2, alpha=0.2
    )
    axis1.plot(
        PSCs.mean(0)[:plot_tmax][::5], label="mean", color="black", lw=lw, zorder=10
    )
    axis1.axis("off")
    axis1.set_title("true parameters")

    axis2.plot(
        PSCs_inferred.T[:plot_tmax][::5],
        label="data",
        color=color["inferred"],
        lw=lw * 0.2,
        alpha=0.2,
    )
    axis2.plot(
        PSCs_inferred.mean(0)[:plot_tmax][::5],
        label="mean",
        color=color["inferred"],
        lw=lw,
        zorder=10,
    )
    axis2.axis("off")
    axis2.set_title("inferred parameters")


def plot_contour(axis, data):
    xparam, yparam = data["variable_parameters"]
    axis.contourf(
        data["xgrid"],
        data["ygrid"],
        data["nll"] / data["nll"].min(),
        levels=[1.005, 1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 5],
        cmap="Greys",
        vmin=0.9,
        vmax=1.6,
    )
    axis.plot(
        true_parameters[xparam], true_parameters[yparam], marker="*", color="black"
    )
    axis.plot(
        data["xgrid"].flatten()[data["nll"].argmin()],
        data["ygrid"].flatten()[data["nll"].argmin()],
        marker="*",
        color=color["inferred"],
    )

    axis.set_xlabel(parameter_names[xparam])
    axis.set_ylabel(parameter_names[yparam])


def plot_MSE(axis):
    data = np.vstack([d["mse"] for d in trainingfits.values()])
    means = data.mean(1)
    sem = scipy.stats.sem(data, 1)

    axis.errorbar(
        training_nspikes,
        means,
        yerr=sem,
        color="black",
        markersize=markersize,
        lw=lw,
        marker="o",
    )
    axis.hlines(
        mse(testtargets, testtargets.mean(0)),
        ls="dashed",
        xmin=np.min(training_nspikes),
        xmax=np.max(training_nspikes),
    )
    axis.set_ylabel("MSE")
    axis.set_xlabel("nr. spikes")
    axis.set_xscale("log")


def plot_paramerror(ax1, ax2):

    ax1.errorbar(
        training_nspikes,
        parameter_errors["sigma_baseline"].mean(0) * 100,
        yerr=scipy.stats.sem(parameter_errors["sigma_baseline"] * 100, 0),
        markersize=markersize,
        lw=lw,
        marker="o",
        color=color["first"],
        label=r"$b_{\sigma}$",
    )
    ax1.errorbar(
        training_nspikes,
        parameter_errors["sigma_amps"].mean(0) * 100,
        yerr=scipy.stats.sem(parameter_errors["sigma_amps"] * 100, 0),
        markersize=markersize,
        lw=lw,
        marker="o",
        color=color["second"],
        label=r"$a_{\sigma}$",
    )
    ax1.errorbar(
        training_nspikes,
        parameter_errors["sigma_scale"].mean(0) * 100,
        yerr=scipy.stats.sem(parameter_errors["sigma_scale"] * 100, 0),
        markersize=markersize,
        lw=lw,
        marker="o",
        color=color["third"],
        label=r"$\sigma_0$",
    )
    ax1.set_xscale("log")
    ax1.legend(frameon=False)
    ax1.set_ylabel(r"error (\%)")
    ax1.set_xlabel("nr. spikes")

    ax2.errorbar(
        training_nspikes,
        parameter_errors["mu_baseline"].mean(0) * 100,
        yerr=scipy.stats.sem(parameter_errors["mu_baseline"] * 100, 0),
        markersize=markersize,
        lw=lw,
        marker="o",
        color=color["first"],
        label=r"$b_{\mu}$",
    )
    ax2.errorbar(
        training_nspikes,
        parameter_errors["mu_amps"].mean(0) * 100,
        yerr=scipy.stats.sem(parameter_errors["mu_baseline"] * 100, 0),
        markersize=markersize,
        lw=lw,
        marker="o",
        color=color["second"],
        label=r"$a_{\mu}$",
    )
    ax2.set_xscale("log")
    ax2.legend(frameon=False)
    ax2.set_xlabel("nr. spikes")


def plot_fig7():

    labels = np.array(
        [
            ["A"] * 6 + ["C"] * 3 + ["E"] * 3,
            ["A"] * 6 + ["C"] * 3 + ["E"] * 3,
            ["B"] * 6 + ["C"] * 3 + ["E"] * 3,
            ["B"] * 6 + ["C"] * 3 + ["E"] * 3,
            ["B"] * 6 + ["D"] * 3 + ["F"] * 3,
            ["B"] * 6 + ["D"] * 3 + ["F"] * 3,
            ["B"] * 6 + ["D"] * 3 + ["F"] * 3,
            ["B"] * 6 + ["D"] * 3 + ["F"] * 3,
            ["G"] * 4 + ["H"] * 4 + ["I"] * 4,
            ["G"] * 4 + ["H"] * 4 + ["I"] * 4,
            ["G"] * 4 + ["H"] * 4 + ["I"] * 4,
            ["G"] * 4 + ["H"] * 4 + ["I"] * 4,
        ]
    )

    fig = MultiPanel(
        labels=labels, figsize=figsize, label_size=10, label_location=(-0.5, 1.15)
    )

    # Split panels B and G
    fig.panels.B.axis("off")

    axB1 = fig.fig.add_subplot(fig.gridspec[2:5, 0:6])
    axB2 = fig.fig.add_subplot(fig.gridspec[5:8, 0:6])

    # PLOTS
    # A
    plot_spiketrain(fig.panels[0], spiketrain[:plot_tmax])
    # B
    plot_PSCs(axB1, axB2)
    # C-F
    plot_contour(fig.panels.C, nll_landscapes["C"])
    plot_contour(fig.panels.D, nll_landscapes["D"])
    plot_contour(fig.panels.E, nll_landscapes["E"])
    plot_contour(fig.panels.F, nll_landscapes["F"])
    fig.panels.C.set_yticks([150, 300, 450])
    fig.panels.D.set_yticks([-3, -2, -1])
    fig.panels.E.set_yticks([-3, -2, -1])
    fig.panels.F.set_yticks([100, 200, 300])

    # G
    plot_paramerror(fig.panels.G, fig.panels.H)

    # I
    plot_MSE(fig.panels.I)

    plt.savefig(current_dir / "figures" / "Fig7_raw.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    plot_fig7()

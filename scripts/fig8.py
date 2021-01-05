# General
import pickle
from pathlib import Path
import os, inspect
import numpy as np
# Models
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from srplasticity.srp import ExpSRP
from srplasticity.inference import fit_srp_model
# Plotting
#from spiffyplots import MultiPanel
import matplotlib.pyplot as plt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

fitting_tm = False
fitting_srp = True
draw_plot = True

current_dir = Path(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)
parent_dir = Path(os.path.dirname(current_dir))

modelfit_dir = current_dir / 'scripts' / "modelfits"
data_dir = current_dir / "data" / "processed" / "chamberland2018"
#modelfit_dir = current_dir / "modelfits"
#data_dir = parent_dir / "data" / "processed" / "chamberland2018"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS
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


def get_model_estimates(model, stimulus_dict):
    """
    :return: Model estimates for training dataset
    """
    estimates = {}

    for key, isivec in stimulus_dict.items():
        if isinstance(model, ExpSRP):
            estimates[key] = model.run_ISIvec(isivec, ntrials=1000)
        else:
            estimates[key] = model.run_ISIvec(isivec)
        model.reset()

    return estimates


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

# Remove in-vivo stimulation as test set
stimulus_dict_test = {"invivo": stimulus_dict.pop("invivo")}
target_dict_test = {"invivo": target_dict.pop("invivo")}


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

    # Fit TM model to data
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


else:
    print("Loading fitted TM model parameters...")
    tm_params = load_pickle(modelfit_dir / "chamberland2018_TMmodel.pkl")

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

    # Parameter ranges
    #srp_param_ranges = (
    #    slice(0.001, 0.0105, 0.0005),  # U
    #    slice(0.001, 0.0105, 0.0005),  # f
    #    slice(1, 501, 10),  # tau_u
    #    slice(1, 501, 10),  # tau_r
    #)

    initial_guess = [2, *mu_kernel_taus, -2, *sigma_kernel_taus, 1]

    # Fit SRP model to data
    srp_params, res = fit_srp_model(initial_guess, stimulus_dict, target_dict,
                        mu_kernel_taus, sigma_kernel_taus,
                        mu_scale=None, bounds=[(-5, 5), *[(-np.inf, np.inf)] * len(mu_kernel_taus),
                                               (-5, 5), *[(-np.inf, np.inf)] * len(sigma_kernel_taus),
                                               (0.001, np.inf)],
                        algo='L-BFGS-B', options={'maxiter': 2000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-9})

    print(res)

    # Save fitted TM model parameters
    save_pickle(srp_params, modelfit_dir / "chamberland2018_SRPmodel.pkl")

else:
    print("Loading fitted SRP model parameters...")
    srp_params = load_pickle(modelfit_dir / "chamberland2018_SRPmodel.pkl")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# GET MODEL ESTIMATES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

tm_train = get_model_estimates(TsodyksMarkramModel(*tm_params), stimulus_dict)
tm_test = get_model_estimates(TsodyksMarkramModel(*tm_params), stimulus_dict_test)

srp_train = get_model_estimates(ExpSRP(*srp_params), stimulus_dict)
srp_test = get_model_estimates(ExpSRP(*srp_params), stimulus_dict_test)

print(srp_test['invivo'][0])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot():
    NotImplemented


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING SCRIPT
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    NotImplemented

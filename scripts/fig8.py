# General
import pickle
from pathlib import Path
import os, inspect

# Models
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from srplasticity.srp import ProbSRP

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

fitting_tm = False
fitting_srp = True
draw_plot = True

current_dir = Path(os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
))
parent_dir = Path(os.path.dirname(current_dir))

modelfit_dir = current_dir / 'modelfits'
data_dir = parent_dir / 'data' / 'processed' / 'chamberland2018'

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
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print('Object pickled and saved.')


def get_model_estimates(model, stimulus_dict):
    """
    :return: Model estimates for training dataset
    """
    estimates = {}

    for key, isivec in stimulus_dict.items():
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
    target_dict[key] = load_pickle(Path(data_dir / str(key + "_normalized_by_cell.pkl")))

# Remove in-vivo stimulation as test set
stimulus_dict_test = {'invivo': stimulus_dict.pop('invivo')}
target_dict_test = {'invivo': target_dict.pop('invivo')}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING TM MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if fitting_tm:
    print('Fitting TM model to Chamberland et al. (2018) data...')

    # Tsodyks-Markram model parameter ranges
    tm_param_ranges = (
        slice(0.001, 0.0105, 0.0005),  # U
        slice(0.001, 0.0105, 0.0005),  # f
        slice(1, 501, 10),  # tau_u
        slice(1, 501, 10),  # tau_r
    )

    # Fit TM model to data
    tm_params, tm_sse, grid, sse_grid = fit_tm_model(stimulus_dict, target_dict, tm_param_ranges,
                                                 disp=True,  # display output
                                                 workers=-1,  # split over all available CPU cores
                                                 full_output=True,  # save function value at each grid node
                                                 )

    # Save fitted TM model parameters
    save_pickle(tm_params, modelfit_dir / 'chamberland2018_TMmodel.pkl')


else:
    print('Loading fitted TM model parameters...')
    tm_params = load_pickle(modelfit_dir / 'chamberland2018_TMmodel.pkl')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING SRP MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if fitting_srp:
    print('Fitting SRP model to Chamberland et al. (2018) data...')

else:
    print('Loading fitted SRP model parameters...')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# GET MODEL ESTIMATES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

tm_train = get_model_estimates(TsodyksMarkramModel(*tm_params), stimulus_dict)
tm_test = get_model_estimates(TsodyksMarkramModel(*tm_params), stimulus_dict_test)

#srp_train = get_model_estimates(SRPmodel(*srp_params), stimulus_dict)
#srp_test = get_model_estimates(SRPmodel(*srp_params), stimulus_dict_test)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
import numpy as np
from pathlib import Path
from src.tools import load_pickle, save_pickle
from src.config import path_processed, path_models

from src.fitting import do_GridSearch, eval_PNfit, Bootstrap_subsets, eval_PNfit_BT
from src.synapses import TMclassic as synapse

import matplotlib.pyplot as plt
import matplotlib
from spiffyplots import MultiPanel

matplotlib.style.use("spiffy")

# # # # # # # # # # # # # #
# October 2020
# Julian Rossbroich
#
# Script for fitting the Tsodyks-Markram STP model to MF-PN synapse data.
#
# PARAMETERS
# # # # # # # # # # # # # #

# - Grid search parameter ranges
parameters = {
    "U": list(np.linspace(0.002, 0.2, 100, endpoint=True)),
    "f": list(np.linspace(0.002, 0.2, 100, endpoint=True)),
    "tau_U": list(np.linspace(2, 500, 250, endpoint=True)),
    "tau_R": [50],  # restricted to 50ms based on calcium model
    "A": [np.nan],
}  # [np.nan] because fitted to normalized data.

# - Save directory
directory = path_models / "2020.10.22_TMmodel"

# - Use Bootstrap?
Bootstrap = False
BT_n = 200
BT_ss = 7
BT_eval = np.mean

# # # # # # # # # # # # # #
# PREPARATION
# # # # # # # # # # # # # #

# generate save directory
if not os.path.exists(directory):
    os.makedirs(directory)

# Stimulation vectors as ISI intervals
ISIpatterns = {
    "20": [0] + [50] * 9,
    "100": [0] + [10] * 9,
    "20100": [0, 50, 50, 50, 50, 10],
    "10020": [0, 10, 10, 10, 10, 50],
    "10100": [0, 100, 100, 100, 100, 10],
    "111": [0] + [9] * 5,
    "invivo": [0, 6, 90.9, 12.5, 25.6, 9],
}

# Loading Data
data = {}
for key in ISIpatterns:
    data[key] = load_pickle(Path(path_processed / str(key + "_normalized_by_cell.pkl")))


# # # # # # # # # # # # # #
# MODEL FITTING
# # # # # # # # # # # # # #

print("STEP 1: GRID SEARCH")
grid, estimates = do_GridSearch(
    ISIpatterns=ISIpatterns, parameters=parameters, synapse=synapse
)
print("GRID SEARCH COMPLETE")


# # # # # # # # # # # # # #
# EVALUATION
# # # # # # # # # # # # # #

print("STEP 2: FINDING BEST SOLUTION...")
signal = np.round(np.arange(0, len(estimates), len(estimates) / 100), 0)
sse = np.zeros(len(estimates))

if Bootstrap == True:
    sse = np.zeros((len(estimates), BT_n))
    BT_data = Bootstrap_subsets(data, BT_ss, BT_n)
    for sol in range(len(estimates)):
        est = estimates[sol]
        sse[sol, ...] = eval_PNfit_BT(est, BT_data, BT_ss=BT_ss)
        if sol in signal:
            print(
                "Calculated {}% of Loss function.".format((sol / len(estimates)) * 100)
            )
    sse_eval = BT_eval(sse, axis=0)

else:
    for sol in range(len(estimates)):
        est = estimates[sol]
        sse[sol] = eval_PNfit(est, data)
        if sol in signal:
            print(
                "Calculated {}% of Loss function.".format((sol / len(estimates)) * 100)
            )
    sse_eval = sse

argbest = np.argmin(sse_eval)
bestsol = {}
bestsol["sse"] = sse_eval[argbest]
bestsol["mse"] = sse_eval[argbest] / np.sum(
    [np.count_nonzero(~np.isnan(x)) for x in data.values()]
)

bestsol["params"] = dict(grid.iloc[argbest])
bestsol["est"] = estimates[argbest]

save_pickle(bestsol, directory / "PN_parameters.pkl")
grid.iloc[np.argsort(sse_eval)[:50]].to_csv(
    directory / "PN_best50.csv", sep=",", index=True, header=True
)

print("FITTING PROCEDURE COMPLETED.")
print("MSE: {}".format(bestsol["mse"]))
print("Parameters: {}".format(bestsol["params"]))

# # # # # # # # # # # # # #
# MAKING PLOTS
# # # # # # # # # # # # # #
protocols = list(ISIpatterns.keys())

# Plot to inspect data
fig = MultiPanel(grid=[len(protocols)], figsize=(20, 4))
panel = 0

for key in protocols:

    # plot for inspection
    fig.panels[panel].set_title(str("protocol:" + str(key)))
    fig.panels[panel].set_ylabel("norm. EPSC")
    fig.panels[panel].set_xlabel("# stim")
    fig.panels[panel].plot(
        range(1, data[key].shape[1] + 1), data[key].T, color="grey", lw=0.2
    )
    fig.panels[panel].plot(
        range(1, data[key].shape[1] + 1),
        np.nanmean(data[key], 0),
        color="black",
        marker="s",
        label="data mean",
    )
    fig.panels[panel].plot(
        range(1, data[key].shape[1] + 1),
        bestsol["est"][key],
        color="darkred",
        marker="o",
        label="TM model",
    )
    fig.panels[panel].set_ylim(0, 10)
    panel += 1

fig.panels[0].legend(frameon=False)
plt.tight_layout()
plt.savefig(directory / "TM_modelfit.pdf")
plt.show()

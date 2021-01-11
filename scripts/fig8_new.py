"""
Generates Fig 8 of the manuscript
"""

# General
import pickle
from pathlib import Path
import os, inspect
import numpy as np
import scipy
import string
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Models
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from srplasticity.srp import ExpSRP, ExponentialKernel
from srplasticity.inference import fit_srp_model, fit_srp_model_gridsearch

# Plotting
from spiffyplots import MultiPanel
import matplotlib.pyplot as plt
import matplotlib

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FIG 8 SCRIPT OPTIONS
# Set these to True to run the parameter inference algorithm.
# Set to False to load fitted parameters from `scripts / modelfits`
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

FITTING_TM = False  # Fit on whole dataset
FITTING_SRP = False  # Fit on whole dataset
DO_BOOTSTRAP = False  # Bootstrap procedure for model comparison

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS: TM MODEL FITTING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LOSS = 'default'  # type of loss function to optimize
HCSE = False  # Whether to use Heteroskedasticity-consistent standard errors (Weighted least sqaures)

TM_PARAM_RANGES = (
    slice(0.001, 0.0105, 0.0005),  # U
    slice(0.001, 0.0105, 0.0005),  # f
    slice(1, 501, 10),  # tau_u
    slice(1, 501, 10),  # tau_r
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS: SRP MODEL FITTING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

MU_KERNEL_TAUS = [15, 100, 650]  # mu kernel time constants
SIGMA_KERNEL_TAUS = [15, 100, 650]  # sigma kernel time constants
SIGMA_SCALE = 4  # Initial guess for sigma scale

# Parameter ranges for grid search. Total of 256 initial starts
SRP_PARAM_RANGES = (
    slice(-3, 1, 0.25),  # both baselines
    slice(-2, 2, 0.25),  # all amplitudes (weighted by tau in fitting procedure)
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS: BOOTSTRAP
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

BOOTSTRAP_N = 20
BOOTSTRAP_SIZE = 0.8  # proportion of data included in each bootstrap

# Set seeds for bootstrap data sampling
BOOTSTRAP_N = np.arange(2020, 2020 + BOOTSTRAP_N, 1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS: CROSSVALIDATION
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Test keys used for crossvalidation (uses all protocols)
TESTKEYS = ["invivo", "100", "20", "20100", "111", "10100", "10020"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PATHS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CURRENT_DIR = Path(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)
PARENT_DIR = Path(os.path.dirname(CURRENT_DIR))

DATA_DIR = PARENT_DIR / "data" / "processed" / "chamberland2018"
FIGURE_DIR = CURRENT_DIR / "figures"
SUPPLEMENT_DIR = CURRENT_DIR / "supplements"
MODELFIT_DIR = CURRENT_DIR / "modelfits"
BOOTSTRAP_DIR = CURRENT_DIR / "bootstrap"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PARAMETERS: PLOTTING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Which testsets to plot for model comparisons
PLOT_TESTKEYS = ["20", "20100", "invivo"]

matplotlib.style.use('spiffy')
matplotlib.rc("xtick", top=False)
matplotlib.rc("ytick", right=False)
matplotlib.rc("ytick.minor", visible=False)
matplotlib.rc("xtick.minor", visible=False)
plt.rc("font", size=8)
#plt.rc("text", usetex=True)

PLOT_FIGSIZE = (5.25102 * 1.5, 5.25102 * 1.5)  # From LaTeX readout of textwidth

PLOT_COLOR = {"tm": "blue",
              "srp": "darkred",
              "kernel": ["#525252", "#969696", "#cccccc"]}

PLOT_MARKERSIZE = 2
PLOT_CAPSIZE = 2
PLOT_LW = 1

PLOT_PROTOCOLNAMES = {
    "100": "10 x 100 Hz",
    "20": "10 x 20 Hz",
    "111": "6 x 111 Hz",
    "20100": "5 x 20 Hz + 1 x 100 Hz",
    "10100": "5 x 10 Hz + 1 x 100 Hz",
    "10020": "5 x 100 Hz + 1 x 20 Hz",
    "invivo": "in-vivo burst",
}

PLOT_PROTOCOLNAMES_SHORT = {
    "100": "100 Hz",
    "20": "20 Hz",
    "111": "111 Hz",
    "20100": "20/100 Hz",
    "10100": "10/100 Hz",
    "10020": "100/20 Hz",
    "invivo": "in-vivo",
}
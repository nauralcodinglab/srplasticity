# Extraction of MF-PN data from .abf files

import os
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib
from spiffyplots import MultiPanel

matplotlib.style.use("spiffy")

from pathlib import Path
from src.data_import import MF_PN_Data
from src.config import path_raw_abfs, path_processed
from src.tools import save_pickle, load_pickle
from scipy.stats.stats import pearsonr

# Options
protocols = list(path_raw_abfs.keys())

# Import Data
dat = MF_PN_Data()
dat.importallcells()


# DEV OF NOISE ANALYSIS
keys = ["20", "100", "20100", "10020", "111", "invivo"]
normalize_by_std = False

fig = MultiPanel(grid=[2], figsize=(6, 3))
panel = 0
for key in ["20", "100", "20100", "10020", "111", "invivo"]:
    paired = dat.noisecor_by_protocol(key, normalized=normalize_by_std)

    fig.panels[panel].hlines(
        y=0, xmax=np.nanmax(paired[:, 0]), xmin=np.nanmin(paired[:, 0]), color="darkred"
    )
    fig.panels[panel].vlines(
        x=0, ymax=np.nanmax(paired[:, 1]), ymin=np.nanmin(paired[:, 1]), color="darkred"
    )
    fig.panels[panel].scatter(paired[:, 0], paired[:, 1], s=2, c="black")
    fig.panels[panel].set_xlabel(r"$\Delta x(s)$ (in SD)")
    fig.panels[panel].set_ylabel(r"$\Delta x(s+1)$ (in SD)")

    panel += 1

plt.show()

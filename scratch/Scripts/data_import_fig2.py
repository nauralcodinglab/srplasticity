import numpy as np
import os, inspect
import pandas as pd
from tools import load_pickle, save_pickle

# DATA IMPORTING
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

xl_12 = pd.read_excel(
    parent_dir + "/Data/data_chamberland_2014.xlsx",
    index_col=0,
    header=None,
    sheet_name="1.2 mM traces with average",
)

traces_12 = np.asarray(xl_12).T[:-1, 1500:3000]
meantrace_12 = np.asarray(xl_12).T[-1, 1500:3000]

xl_25 = pd.read_excel(
    parent_dir + "/Data/data_chamberland_2014.xlsx",
    index_col=0,
    header=None,
    sheet_name="2.5 mM traces with average",
)

traces_25 = np.asarray(xl_25).T[:-1, 1500:3000]
meantrace_25 = np.asarray(xl_25).T[-1, 1500:3000]

traces = {"1.2ca": traces_12, "2.5ca": traces_25}

amps = pd.read_excel(
    parent_dir + "/Data/data_chamberland_2014.xlsx",
    index_col=0,
    header=0,
    sheet_name="middle plot",
)
cvs = pd.read_excel(
    parent_dir + "/Data/data_chamberland_2014.xlsx",
    index_col=0,
    header=0,
    sheet_name="right plot",
)

save_pickle(amps.to_dict("list"), parent_dir + "/Data/fig2_amps.pkl")
save_pickle(cvs.to_dict("list"), parent_dir + "/Data/fig2_cvs.pkl")
save_pickle(traces, parent_dir + "/Data/fig2_traces.pkl")

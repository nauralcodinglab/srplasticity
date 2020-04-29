import numpy as np
import os, inspect
import pandas as pd
from tools import load_pickle

# DATA IMPORTING
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
data_fig2 = load_pickle(parent_dir + '/Data/data_fig2.pkl')

xl_12 = pd.read_excel(parent_dir + '/Data/data_chamberland_2014.xlsx', index_col=0, header=1,
                   sheet_name='1.2 mM traces with average')
traces_12 = np.asarray(xl_12).T[:-1, 1500:3000]
meantrace_12 = np.asarray(xl_12).T[-1, 1500:3000]

xl_25 = pd.read_excel(parent_dir + '/Data/data_chamberland_2014.xlsx', index_col=0, header=1,
                   sheet_name='2.5 mM traces with average')
traces_25 = np.asarray(xl_25).T[:-1, 1500:3000]
meantrace_25 = np.asarray(xl_25).T[-1, 1500:3000]
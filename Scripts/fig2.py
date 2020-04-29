import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools import add_figure_letters


# PLOT SETTINGS
# # # # # # # # # #
plt.style.use('science')
matplotlib.rc('xtick', top=False)
matplotlib.rc('ytick', right=False)
matplotlib.rc('ytick.minor', visible=False)
matplotlib.rc('xtick.minor', visible=False)
matplotlib.rc('axes.spines', top=False, right=False)
plt.rc('font', size=8)

# plt.rc('text', usetex=False)
# plt.rc('font', family='sans-serif')

markersize = 3
lw = 1
figsize = (5.25102, 5.25102)  # From LaTeX readout of textwidth

c_13ca = '#cc3311'
c_25ca = 'black'
c_params = ['#000000', '#BBBBBB']


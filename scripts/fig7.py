
# Plotting
from spiffyplots import MultiPanel
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PLOTTING OPTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Which testsets to plot for model comparisons
plotted_testsets_ordered = ["20", "20100", "invivo"]

plt.style.use("spiffy")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rc("font", size=7, family="serif")
plt.rc("text", usetex=True)

figsize = (5.25102, 5.25102 * 1.05)  # From LaTeX readout of textwidth

color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
c_kernels = ("#525252", "#969696", "#cccccc")  # greyscale

markersize = 3
capsize = 2
lw = 1
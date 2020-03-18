# GLOBAL DEPENDENCIES
import pylab as pl

# FIGURE 1
# # # # # # # # # #

from Scripts.fig1 import plot as plot_fig1

figure1, figure1_spiketrain = plot_fig1()
figure1.savefig('Figures/Fig1.pdf')
figure1_spiketrain.savefig('Figures/Fig1_spiketrain.pdf')
pl.close('all')

# FIGURE 2
# # # # # # # # # #

from Scripts.fig2 import plot as plot_fig2

fig2_data, fig2_gaus, fig2_model, fig2_steps = plot_fig2()

fig2_data.savefig('Figures/Fig2_data.pdf')
fig2_gaus.savefig('Figures/Fig2_gaussians.pdf')
fig2_model.savefig('Figures/Fig2_modelres.pdf')
fig2_steps.savefig('Figures/Fig2_modelsteps.pdf')
pl.close('all')

# FIGURE 3
# # # # # # # # # #

# FIGURE 4
# # # # # # # # # #

# FIGURE 5
# # # # # # # # # #

# FIGURE 6
# # # # # # # # # #

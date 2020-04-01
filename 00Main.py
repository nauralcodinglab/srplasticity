# GLOBAL DEPENDENCIES
import pylab as pl

# FIGURE 1
# # # # # # # # # #

from Scripts.fig1 import plot as plot_fig1

figure1, figure1_spiketrain = plot_fig1()
figure1.savefig('Figures/Fig1.pdf')
figure1_spiketrain.savefig('Figures/Fig1_spiketrain.pdf')
pl.close('all')


# FIGURE 3 - MODELS
# # # # # # # # # #



# FIGURE 4
# # # # # # # # # #

from Scripts.fig4 import plot as plot_fig4

fig4_data, fig4_gaus, fig4_model, fig4_steps = plot_fig4()

fig4_data.savefig('Figures/Fig4_data.pdf')
fig4_gaus.savefig('Figures/Fig4_gaussians.pdf')
fig4_model.savefig('Figures/Fig4_modelres.pdf')
fig4_steps.savefig('Figures/Fig4_modelsteps.pdf')
pl.close('all')

# FIGURE 5
# # # # # # # # # #

# FIGURE 6
# # # # # # # # # #

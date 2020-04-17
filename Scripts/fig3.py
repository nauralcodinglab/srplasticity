from eTMmodel import cTMSynapse, gTMSynapse
from LNLmodel import det_exp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools import get_stimvec, add_figure_letters
import scipy.stats as stats

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

# FITTED MODEL PARAMETERS
# # # # # # # # # #

g_MF_PN = gTMSynapse(1,
                     U=np.array([0.207]),
                     f=np.array([0.545]),
                     tau_U=np.array([500]),
                     tau_R=np.array([50]),
                     Z=np.array([0.207]),
                     tau_Z=np.array([7.531]),
                     w=1 / (0.207 * 0.207))

c_MF_PN = cTMSynapse(1,
                     U=np.array([0.02]),
                     f=np.array([0.02]),
                     tau_U=np.array([130]),
                     tau_R=np.array([50]),
                     w=1 / (0.02))

# PARAMETERS cTM EXAMPLE
# # # # # # # # # # # # #
syn_c_13 = cTMSynapse(1,
                      U=np.array([0.05]),
                      f=np.array([0.2]),
                      tau_U=np.array([50]),
                      tau_R=np.array([10]))
syn_c_25 = cTMSynapse(1,
                      U=np.array([0.25]),
                      f=np.array([0.2]),
                      tau_U=np.array([50]),
                      tau_R=np.array([10]))

# PARAMETERS gTM EXAMPLE
# # # # # # # # # # # # #
syn_g_13 = gTMSynapse(1,
                   U=np.array([0.05]),
                   f=np.array([1]),
                   tau_U=np.array([50]),
                   tau_R=np.array([10]),
                   Z=np.array([1]),
                   tau_Z=np.array([1]))

syn_g_25 = gTMSynapse(1,
                   U=np.array([0.25]),
                   f=np.array([1]),
                   tau_U=np.array([50]),
                   tau_R=np.array([10]),
                   Z=np.array([1]),
                   tau_Z=np.array([1]))

# PARAMETERS LNL EXAMPLE
# # # # # # # # # # # # #

syn_lnl_25 = det_exp(a=600,
                   tau=500,
                   b=-1.5,
                   T=None)

syn_lnl_13 = det_exp(a=600,
                   tau=500,
                   b=-5.5,
                   T=None)


# EXAMPLE SPIKES
# # # # # # # # # # # # #
dt = 0.1
nrspikes = 5
isi = 20
examplespikes = np.array([get_stimvec([isi] * nrspikes, dt=dt, null=5, extra=30)])


def plot():
    # Make Figure Grid
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, wspace=0.1, hspace=0.1)
    axA1 = fig.add_subplot(spec[0, 0])
    axA2 = fig.add_subplot(spec[0, 1])
    axA3 = fig.add_subplot(spec[0, 2])
    axB1 = fig.add_subplot(spec[1, 0])
    axB2 = fig.add_subplot(spec[1, 1])
    axB3 = fig.add_subplot(spec[1, 2])
    axC1 = fig.add_subplot(spec[2, 0])
    axC2 = fig.add_subplot(spec[2, 1])
    axC3 = fig.add_subplot(spec[2, 2])

    # Make plots
    axA1, axA2 = plot_cTM_mech(axA1, axA2)
    axA3 = plot_cTM_eff(axA3)
    axB1, axB2 = plot_gTM_mech(axB1, axB2)
    axB3 = plot_gTM_eff(axB3)
    axC1, axC2, axC3 = plot_lnl(axC1, axC2, axC3)

    add_figure_letters([axA1, axA3, axB1, axB3, axC1, axC3], size=12)

    return fig


def plot_cTM_mech(ax1, ax2):
    res_c = syn_c_13.run(examplespikes, dt, update_all=True)
    ax1.plot(res_c['t'], res_c['u'][0], color=c_params[0], lw=lw, label='u')
    ax1.plot(res_c['t'], res_c['R'][0], color=c_params[1], lw=lw, label='R')
    ax1.axhline(y=res_c['u'][0][0], c=c_13ca, ls='dashed', lw=lw)
    ax1.set_xlabel('time (ms)')
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=0)
    ax1.set_yticks([res_c['u'][0][0], 1])
    ax1.set_yticklabels(['U', 1])
    ax1.get_yticklabels()[0].set_color(c_13ca)
    ax1.legend(frameon=False)

    res_c = syn_c_25.run(examplespikes, dt, update_all=True)
    ax2.plot(res_c['t'], res_c['u'][0], color=c_params[0], lw=lw, label='u')
    ax2.plot(res_c['t'], res_c['R'][0], color=c_params[1], lw=lw, label='R')
    ax2.axhline(y=res_c['u'][0][0], c=c_25ca, ls='dashed', lw=lw)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylim(bottom=0, top=1.05)
    ax2.set_xlim(left=0)
    ax2.set_yticks([res_c['u'][0][0], 1])
    ax2.set_yticklabels(['U', 1])
    ax2.get_yticklabels()[0].set_color(c_25ca)
    return ax1, ax2


def plot_cTM_eff(ax):
    res_c = syn_c_13.run(examplespikes, dt, update_all=True)
    efficacy_c = res_c['psr'][0][res_c['psr'][0] > 0]

    ax.plot(np.arange(1, nrspikes + 1), efficacy_c, color=c_13ca, lw=lw, marker='o', markersize=markersize)
    ax.set_xlabel('Spike number')

    res_c = syn_c_25.run(examplespikes, dt, update_all=True)
    efficacy_c = res_c['psr'][0][res_c['psr'][0] > 0]

    ax.plot(np.arange(1, nrspikes + 1), efficacy_c, color=c_25ca, lw=lw, marker='o', markersize=markersize)
    ax.set_xlabel('Spike number')

    ax.xaxis.set_ticks(np.arange(1, 6))
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel('Efficacy')
    ax.yaxis.set_ticks([0, 0.3, 0.6, 0.9])

    return ax


def plot_gTM_mech(ax1, ax2):
    res_c = syn_g_13.run(examplespikes, dt, update_all=True)
    ax1.plot(res_c['t'], res_c['u'][0], color=c_params[0], lw=lw, label='u')
    ax1.plot(res_c['t'], res_c['R'][0], color=c_params[1], lw=lw, label='R')
    ax1.axhline(y=res_c['u'][0][0], c=c_13ca, ls='dashed', lw=lw)
    ax1.set_xlabel('time (ms)')
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=0)
    ax1.set_yticks([res_c['u'][0][0], 1])
    ax1.set_yticklabels(['U', 1])
    ax1.get_yticklabels()[0].set_color(c_13ca)
    ax1.legend(frameon=False)

    res_c = syn_g_25.run(examplespikes, dt, update_all=True)
    ax2.plot(res_c['t'], res_c['u'][0], color=c_params[0], lw=lw, label='u')
    ax2.plot(res_c['t'], res_c['R'][0], color=c_params[1], lw=lw, label='R')
    ax2.axhline(y=res_c['u'][0][0], c=c_25ca, ls='dashed', lw=lw)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylim(bottom=0, top=1.05)
    ax2.set_xlim(left=0)
    ax2.set_yticks([res_c['u'][0][0], 1])
    ax2.set_yticklabels(['U', 1])
    ax2.get_yticklabels()[0].set_color(c_25ca)
    return ax1, ax2

def plot_gTM_eff(ax):
    res_c = syn_g_13.run(examplespikes, dt, update_all=True)
    efficacy_c = res_c['psr'][0][res_c['psr'][0] > 0]

    ax.plot(np.arange(1, nrspikes + 1), efficacy_c, color=c_13ca, lw=lw, marker='o', markersize=markersize)
    ax.set_xlabel('Spike number')

    res_c = syn_g_25.run(examplespikes, dt, update_all=True)
    efficacy_c = res_c['psr'][0][res_c['psr'][0] > 0]

    ax.plot(np.arange(1, nrspikes + 1), efficacy_c, color=c_25ca, lw=lw, marker='o', markersize=markersize)
    ax.set_xlabel('Spike number')

    ax.xaxis.set_ticks(np.arange(1, 6))
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel('Efficacy')
    ax.yaxis.set_ticks([0, 0.3, 0.6, 0.9])

    return ax

def plot_lnl(ax1,ax2,ax3):

    res13 = syn_lnl_13.run(examplespikes[0])
    res25 = syn_lnl_25.run(examplespikes[0])
    t = np.arange(0, examplespikes.shape[1] * dt, dt)

    ax1.plot(t, res13['filtered_s'], color=c_13ca, lw=lw)
    ax1.plot(t, res25['filtered_s'], color=c_25ca, lw=lw)
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel('time (ms)')
    ax1.set_title(r'$\mathbf{k}_\mu\ast S+b$')

    ax2.plot(t, res13['nl_readout'], color=c_13ca, lw=lw)
    ax2.plot(t, res25['nl_readout'], color=c_25ca, lw=lw)
    ax2.set_xlabel('time (ms)')
    ax2.set_title(r'$f(\mathbf{k}_\mu\ast S+b)$')

    eff13 = res13['efficacy'][np.argwhere(res13['efficacy'] != 0)]
    eff25 = res25['efficacy'][np.argwhere(res25['efficacy'] != 0)]
    ax3.plot(np.arange(1, nrspikes + 1), eff13, color=c_13ca, lw=lw, marker='o', markersize=markersize)
    ax3.plot(np.arange(1, nrspikes + 1), eff25, color=c_25ca, lw=lw, marker='o', markersize=markersize)
    ax3.set_xlabel('Spike number')
    ax3.xaxis.set_ticks(np.arange(1, 6))
    ax3.set_ylim(bottom=0, top=1)
    ax3.set_ylabel('Efficacy')
    ax3.yaxis.set_ticks([0, 0.3, 0.6, 0.9])

    return ax1, ax2, ax3

if __name__ == '__main__':
    import os, inspect

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    fig.savefig(parent_dir + '/Figures/Fig3_raw.pdf')
    plt.show()

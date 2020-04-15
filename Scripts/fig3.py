from eTMmodel import cTMSynapse, gTMSynapse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tools import get_stimvec, cm2inch
from tools import data_fig2
import scipy.stats as stats

# PLOTTING OPTIONS
# # # # # # # # # # # # #

markersize = 4
capsize = 2
cols = {'u': 'black',
        'R': 'darkgray',
        'gTM': '#e31f26',  # blue alternative: 387fb9
        'cTM': '#e31f26',
        'lnl': '#e31f26',
        'data': 'black'}

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=8)

figsize_mech = cm2inch(4.5, 5)
figsize_eff = cm2inch(3.5, 5)
figsize_fit = cm2inch(4, 5)
figsize_ppr = cm2inch(3.5, 5)

# FITTED MODEL PARAMETERS

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
syn_c = cTMSynapse(1,
                   U=np.array([0.04]),
                   f=np.array([0.15]),
                   tau_U=np.array([100]),
                   tau_R=np.array([20]))

# PARAMETERS gTM EXAMPLE
# # # # # # # # # # # # #
syn_g = gTMSynapse(1,
                   U=np.array([0.04]),
                   f=np.array([0.7]),
                   tau_U=np.array([100]),
                   tau_R=np.array([20]),
                   Z=np.array([1]),
                   tau_Z=np.array([1]))

# EXAMPLE SPIKES
# # # # # # # # # # # # #
dt = 0.1
nrspikes = 10
isi = 10
examplespikes = np.array([get_stimvec([isi] * nrspikes, dt=dt, null=5, extra=30)])

# LOAD DATA
data = data_fig2['100'][..., :6]  # 100Hz data
data = np.concatenate([data, data_fig2['longstim'][..., :6]])  # append longstim
for n in range(len(data)):  # normalize EPSCs
    data[n] = data[n] / data[n][0]
sem = stats.sem(data, nan_policy='omit')  # errorbars
spikes = np.array([get_stimvec([10] * 6, dt=dt, null=5, extra=5)])

# Panel A: Mechanism and efficacy of the cTM model
# # # # # # # # # # # # #

# RUN MODEL MECHANISM EXAMPLE
res_c = syn_c.run(examplespikes, dt, update_all=True)
efficacy_c = res_c['psr'][0][res_c['psr'][0] > 0]
normefficacy_c = efficacy_c / efficacy_c[0]

res_g = syn_g.run(examplespikes, dt, update_all=True)
efficacy_g = res_g['psr'][0][res_g['psr'][0] > 0]
normefficacy_g = efficacy_g / efficacy_g[0]


def plot_cTM_mech():
    fig, ax = plt.subplots(figsize=figsize_mech)
    ax.plot(res_c['t'], res_c['u'][0], color=cols['u'], lw=0.7)
    ax.plot(res_c['t'], res_c['R'][0], color=cols['R'], lw=0.7)
    ax.set_xlabel('time (ms)')
    ax.set_ylim(bottom=0, top=1.05)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Hide Y axis ticks and tickmarks
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


def plot_cTM_eff():
    fig, ax = plt.subplots(figsize=figsize_eff)
    ax.plot(np.arange(1, nrspikes + 1), normefficacy_c, color=cols['cTM'], lw=0.7, marker='s', markersize=markersize)
    ax.set_xlabel('Spike number')
    ax.xaxis.set_ticks([1, 4, 7, 10])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Hide Y axis ticks and tickmarks
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


def plot_gTM_mech():
    fig, ax = plt.subplots(figsize=figsize_mech)
    ax.plot(res_g['t'], res_g['u'][0], color=cols['u'], lw=0.7)
    ax.plot(res_g['t'], res_g['R'][0], color=cols['R'], lw=0.7)
    ax.set_xlabel('time (ms)')
    ax.set_ylim(bottom=0, top=1.05)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Hide Y axis ticks and tickmarks
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


def plot_gTM_eff():
    fig, ax = plt.subplots(figsize=figsize_eff)
    ax.plot(np.arange(1, nrspikes + 1), normefficacy_g, color=cols['gTM'], lw=0.7, marker='s', markersize=markersize)
    ax.set_xlabel('Spike number')
    ax.xaxis.set_ticks([1, 4, 7, 10])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Hide Y axis ticks and tickmarks
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


def plot_gTM_baselines():
    U_range = np.arange(0.05, 0.4, 0.01)
    normeffic = np.zeros([U_range.size,nrspikes])
    colors = cm.Blues(np.linspace(0.25,0.75,U_range.size))
    for i in range(U_range.size):
        syn = gTMSynapse(1,
                         U=np.array([U_range[i]]),
                         f=np.array([0.7]),
                         tau_U=np.array([20]),
                         tau_R=np.array([100]),
                         Z=np.array([1]),
                         tau_Z=np.array([1]))
        res_g = syn.run(examplespikes, dt, update_all=True)
        efficacy_g = res_g['psr'][0][res_g['psr'][0] > 0]
        normeffic[i] = efficacy_g #/ efficacy_g[0]
        plt.plot(normeffic[i], color=colors[i])

    plt.show()

if __name__ == '__main__':
    import os, inspect

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)


def zero():
    # Panel C: Model fit of the cTM model
    # # # # # # # # # # # # # #

    # LOAD DATA
    data = load_pickle('Data/data_fig2.pkl')['100'][..., :6]  # 100Hz data
    data = np.concatenate([data, load_pickle('Data/data_fig2.pkl')['longstim'][..., :6]])  # append longstim
    for n in range(len(data)):  # normalize EPSCs
        data[n] = data[n] / data[n][0]
    sem = stats.sem(data, nan_policy='omit')  # errorbars
    PPR = np.array([[x[i] / x[i - 1] for i in np.arange(1, 6)] for x in data])  # PPR
    PPR_sem = stats.sem(PPR, nan_policy='omit')  # errorbars

    spikes = np.array([get_stimvec([10] * 6, dt=dt, null=5, extra=5)])

    # GET MODEL
    res_c = c_MF_PN.run(spikes, 0.1)
    model_c = res_c['psr'][0][res_c['psr'][0] > 0]
    PPR_c = np.array([model_c[i] / model_c[i - 1] for i in np.arange(1, 6)])

    # PLOT EPSC Amplitudes
    fig, ax = plt.subplots(figsize=figsize_fit)
    ax.plot(np.arange(1, 7), model_c, color=cols['cTM'], lw=0.7, marker='s', markersize=markersize, zorder=2,
            label='model')
    ax.errorbar(np.arange(1, 7), data.mean(0), yerr=sem, color=cols['data'], lw=0.7, marker='s',
                markersize=markersize, capsize=capsize, zorder=1, label='data')
    ax.set_xlabel('Spike number')
    ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6])
    ax.set_ylabel('nEPSC Amplitude')
    ax.set_ylim(bottom=0.8, top=5)
    ax.yaxis.set_ticks([1, 2, 3, 4, 5])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('thesis/fig2/cTM_fit.pdf')
    plt.show()

    # PLOT PPR
    fig, ax = plt.subplots(figsize=figsize_fit)
    ax.plot(np.arange(1, 6), PPR_c, color=cols['cTM'], lw=0.7, marker='s', markersize=markersize, zorder=2,
            label='model')
    ax.errorbar(np.arange(1, 6), PPR.mean(0), yerr=PPR_sem, color=cols['data'], lw=0.7, marker='s',
                markersize=markersize, capsize=capsize, zorder=1, label='data')
    ax.set_xlabel("Spike")
    ax.xaxis.set_ticks([1, 2, 3, 4, 5])
    ax.set_ylabel('PPR')
    ax.set_ylim(bottom=1, top=2)
    ax.yaxis.set_ticks([1.2, 1.4, 1.6, 1.8, 2])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('thesis/fig2/cTM_ppr.pdf')
    plt.show()

    # Panel D: Model fit of the gTM model
    # # # # # # # # # # # # # #

    # GET MODEL
    res_g = g_MF_PN.run(spikes, 0.1)
    model_g = res_g['psr'][0][res_g['psr'][0] > 0]
    PPR_g = np.array([model_g[i] / model_g[i - 1] for i in np.arange(1, 6)])

    # PLOT EPSC Amplitudes
    fig, ax = plt.subplots(figsize=figsize_fit)
    ax.plot(np.arange(1, 7), model_g, color=cols['gTM'], lw=0.7, marker='s', markersize=markersize, zorder=2,
            label='model')
    ax.errorbar(np.arange(1, 7), data.mean(0), yerr=sem, color=cols['data'], lw=0.7, marker='s',
                markersize=markersize, capsize=capsize, zorder=1, label='data')
    ax.set_xlabel('Spike number')
    ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6])
    ax.set_ylabel('nEPSC Amplitude')
    ax.set_ylim(bottom=0.8, top=5)
    ax.yaxis.set_ticks([1, 2, 3, 4, 5])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('thesis/fig2/gTM_fit.pdf')
    plt.show()

    # PLOT PPR
    fig, ax = plt.subplots(figsize=figsize_fit)
    ax.plot(np.arange(1, 6), PPR_g, color=cols['gTM'], lw=0.7, marker='s', markersize=markersize, zorder=2,
            label='model')
    ax.errorbar(np.arange(1, 6), PPR.mean(0), yerr=PPR_sem, color=cols['data'], lw=0.7, marker='s',
                markersize=markersize, capsize=capsize, zorder=1, label='data')
    ax.set_xlabel("Spike")
    ax.xaxis.set_ticks([1, 2, 3, 4, 5])
    ax.set_ylabel('PPR')
    ax.set_ylim(bottom=1, top=2)
    ax.yaxis.set_ticks([1.2, 1.4, 1.6, 1.8, 2])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('thesis/fig2/gTM_ppr.pdf')
    plt.show()

    print('MSE cTM fit:')
    print(np.nansum((data - np.array([model_c] * len(data))) ** 2) / np.size(data))
    print('MSE gTM fit:')
    print(np.nansum((data - np.array([model_g] * len(data))) ** 2) / np.size(data))

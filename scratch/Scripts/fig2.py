import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools import add_figure_letters, load_pickle, add_scalebar_ES


# PLOT SETTINGS
# # # # # # # # # #

plt.style.use('science')
matplotlib.rc('xtick', top=False)
matplotlib.rc('ytick', right=False)
matplotlib.rc('ytick.minor', visible=False)
matplotlib.rc('xtick.minor', visible=False)
matplotlib.rc('axes.spines', top=False, right=False)
plt.rc('font', size=8)


markersize = 3
capsize = 2
lw = 1
lw_traces = 0.5
figsize = (5.25102, 5.25102 * 1/3)  # From LaTeX readout of textwidth

# COLORSCHEME VIBRANT: RED/BLUE
c_12ca = '#cc3311'
c_12ca_trace = '#f9c7bb'
c_25ca = '#0077bb'
c_25ca_trace = '#bbe6ff'

## COLORSCHEME VIBRANT: ORANGE/BLUE
#c_12ca = '#ee7733'
#c_12ca_trace = '#F9D6C1'
#c_25ca = '#0077bb'
#c_25ca_trace = '#bbe6ff'

# COLORSCHEME HIGH CONTRAST: YELLOW/BLUE
#c_12ca = '#ddaa33'
#c_12ca_trace = '#f6e9ca'
#c_25ca = '#004488'
#c_25ca_trace = '#c3e1ff'

# COLORSCHEME BRIGHT: RED/BLUE
#c_12ca = '#ee6677'
#c_12ca_trace = '#f8bec5'
#c_25ca = '#4477aa'
#c_25ca_trace = '#c9d9e9'

def plot():
    # Make Figure Grid
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=10, nrows=1, figure=fig, wspace=0.1, hspace=0.1)
    axLeft = fig.add_subplot(spec[0, 0:3])
    subspec1 = spec[0, 1:4].subgridspec(ncols=1, nrows=2, wspace=0.0, hspace=0.0)
    axTrace1 = fig.add_subplot(subspec1[0, 0])
    axTrace2 = fig.add_subplot(subspec1[1, 0])
    axAmp = fig.add_subplot(spec[0, 4:7])
    axCV = fig.add_subplot(spec[0, 7:])

    axLeft.axis('off')
    axTrace1, axTrace2 = plot_traces(axTrace1, axTrace2)
    axAmp = plot_amps(axAmp)
    axCV = plot_cv(axCV)

    add_figure_letters([axLeft, axAmp, axCV], size=12)

    return fig

def plot_traces(ax1, ax2):

    xax = np.arange(0,traces['1.2ca'].mean(0).shape[0]*0.1,0.1)
    for trace in traces['1.2ca']:
        ax1.plot(xax[::2], trace[::2], color=c_12ca_trace, lw = lw_traces)
    ax1.plot(xax[::2], traces['1.2ca'].mean(0)[::2], color=c_12ca, lw = lw_traces)
    ax1.set_xticks([])
    ax1.set_ylim(bottom=np.min(traces['1.2ca']), top=50)
    ax1.set_title('1.2 mM $[Ca^{2+}]$', color = c_12ca,
                  fontdict={'fontsize': 8})
    add_scalebar_ES(x_units='ms', y_units='pA', anchor=(0.3, 0.1),
                    x_size=20, y_size=100, y_label_space=0.02, x_label_space=-0.1,
                    bar_space=0, x_on_left=False, linewidth=0.5, remove_frame=True,
                    omit_x=False, omit_y=False, round=True, usetex=True, ax=ax1)

    for trace in traces['2.5ca']:
        ax2.plot(xax[::2], trace[::2], color=c_25ca_trace, lw = lw_traces)
    ax2.plot(xax[::2], traces['2.5ca'].mean(0)[::2], color=c_25ca, lw = lw_traces)
    ax2.set_xticks([])
    ax2.set_ylim(bottom=np.min(traces['2.5ca']), top=50)
    ax2.set_title('2.5 mM $[Ca^{2+}]$',color = c_25ca,
                  fontdict={'fontsize': 8})
    add_scalebar_ES(x_units='ms', y_units='pA', anchor=(0.3, 0.15),
                    x_size=20, y_size=500, y_label_space=0.02, x_label_space=-0.1,
                    bar_space=0, x_on_left=False, linewidth=0.5, remove_frame=True,
                    omit_x=False, omit_y=False, round=True, usetex=True, ax=ax2)

    return ax1, ax2

def plot_amps(ax):

    x_ax = np.arange(1,6)
    ax.errorbar(x_ax, amps['1.2ca'], amps['1.2ca_error'],
                capsize=capsize, marker='s', lw=lw, elinewidth=lw * 0.7, markersize=markersize,
                color=c_12ca)
    ax.errorbar(x_ax, amps['2.5ca'], amps['2.5ca_error'],
                capsize=capsize, marker='s', lw=lw, elinewidth=lw * 0.7, markersize=markersize,
                color=c_25ca)
    ax.set_ylim(bottom=0, top =1000)
    ax.set_yticks([0, 500, 1000])
    ax.set_xticks(x_ax)
    ax.set_ylabel('EPSC amplitude (pA)')
    ax.set_xlabel(r'stimulus nr.')

    return ax


def plot_cv(ax):

    x_ax = np.arange(1,6)
    ax.errorbar(x_ax, cvs['1.2ca'], cvs['1.2ca_error'],
                capsize=capsize, marker='s', lw=lw, elinewidth=lw * 0.7, markersize=markersize,
                color=c_12ca)
    ax.errorbar(x_ax, cvs['2.5ca'], cvs['2.5ca_error'],
                capsize=capsize, marker='s', lw=lw, elinewidth=lw * 0.7, markersize=markersize,
                color=c_25ca)
    ax.set_ylim(bottom=0.2, top =0.8)
    ax.set_xticks(x_ax)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_ylabel('CV')
    ax.set_xlabel(r'stimulus nr.')

    return ax

if __name__ == '__main__':
    import os, inspect

    # FIGURE 1 PLOTTING

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)

    # DATA IMPORT
    # # # # # # # # # #
    amps = load_pickle(parent_dir + '/Data/fig2_amps.pkl')
    cvs = load_pickle(parent_dir + '/Data/fig2_cvs.pkl')
    traces = load_pickle(parent_dir + '/Data/fig2_traces.pkl')

    fig = plot()
    plt.tight_layout()
    fig.savefig(parent_dir + '/Figures/Fig2_raw.pdf')
    plt.show()
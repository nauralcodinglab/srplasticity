#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Code for reproducing Figure 6
"""

from pylab import *
from numpy import *
import string
import seaborn as sns
from numpy import loadtxt
from scipy.signal import lfilter
from scipy import special

# PLOT SETTINGS
# # # # # # # # # #

plt.style.use("spiffy")
matplotlib.rc("xtick", top=False)
matplotlib.rc("ytick", right=False)
matplotlib.rc("ytick.minor", visible=False)
matplotlib.rc("xtick.minor", visible=False)
matplotlib.rc("axes.spines", top=False, right=False)
plt.rc("font", size=8)
#plt.rc('text', usetex=False)
#plt.rc('font', family='sans-serif',size=8)

markersize, lw = 3, 1
figsize = (10,8) #(5.25102, 5.25102 * 0.75)  # From LaTeX readout of textwidth

# COLORS
# # # # # # # # # #
c_13ca = "#cc3311"
c_13ca_trace = "#f9c7bb"
c_25ca = "#0077bb"
c_25ca_trace = "#bbe6ff"
c_traces = "lightgrey"

# FUNCTIONS
# # # # # # # # # #

def add_figure_letters(axes, size=14):
    """
    Function to add Letters enumerating multipanel figures.

    :param axes: list of matplotlib Axis objects to enumerate
    :param size: Font size
    """

    for n, ax in enumerate(axes):

        ax.text(
            -0.15,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=size,
            weight="bold",
            usetex=False,
            family="sans-serif",
        )


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + exp(-x))

def exec_func(a,a2,b,b2,tauk):
    '''Initialize kernels and find PSC'''
    spktr = zeros(t.shape)
    k = 1/tauk*exp(-t/tauk)
    sigfac = 0.45 * mufac * sigmoid(b) / sigmoid(b2)
    spktr[[4000,4100,4200,4300,4400]] = 1
    filtered_S, filtered_S2 = a*lfilter(k,1,spktr)+b, a2*lfilter(k,1,spktr)+b2
    filtered_S, filtered_S2 = roll(filtered_S,1), roll(filtered_S2,1)
    if nlin=='sigmoid':
        mean_om = mufac*sigmoid(filtered_S)
        sig = sigfac*sigmoid(filtered_S2)
    else:
        mean_om = mufac*(tanh(filtered_S)+1)/2.0
        sig = sigfac

    I = zeros((Ntrial, len(spktr)))
    for trial in range(Ntrial):
        wspktr, gspktr = mean_om*spktr, sig*spktr
        for ind in where(wspktr>0)[0]:
            if model == 'gamma':
                wspktr[ind] = random.gamma(wspktr[ind]**2/gspktr[ind]**2,scale=gspktr[ind]**2/wspktr[ind])
        tau_psc = 5.0
        kappa=-1*exp(-t/tau_psc)
        I[trial,:] = lfilter(kappa,1,wspktr) + 0.05*random.randn(len(spktr))
    return wspktr, I, mean_om, sig, k

def plot_kernels(axes1):
    axes1[0].plot(append(-t[::-1], t), append(k1 * 0 + b, k1 * a + b), "k", color=c_25ca)
    axes1[0].plot(append(-t[::-1], t), append(k2 * 0 + b3, k2 * a + b3), color=c_13ca)
    axes1[0].set_ylabel(r"$\mu$-kernel")
    axes1[0].set_xlabel("time (ms)")

    axes1[1].plot(append(-t[::-1], t), append(k1 * 0 + b2, k1 * a2 + b2), "k")
    axes1[1].set_yticks([-1, 0])
    axes1[1].set_ylim(bottom=-1.8, top=0.8)
    axes1[1].set_ylabel(r"$\sigma$-kernel")
    axes1[1].set_xlabel("time (ms)")

    return axes1

def plot_traces(axB):
    for trial in range(Ntrial):
        axB.plot(t, I1[trial, :], ".6", lw=0.5, color=c_traces)
        axB.plot(t, I2[trial, :] - 8, ".6", lw=0.5, color=c_traces)
    axB.plot(t, mean(I1, axis=0), "k", lw=lw, color=c_25ca)
    axB.plot(t, mean(I2, axis=0) - 8, "r", lw=1, color=c_13ca)
    axB.plot(array([460, 470]), -array([5, 5]), "k", lw=1.5, solid_capstyle="butt")      
    axB.set_xlim((390, 470))
    axB.axis("off")
    axB.set_ylim(bottom=-12, top=3)
    axB.set_xlabel([])  
    
    return axB

def plot_mu_sig(axes2):
    munorm1 = mean_om1[where(wspktr1 > 0)[0][0]]
    
    axes2[0].plot(range(1, 6), mean_om1[where(wspktr1 > 0)[0]] / munorm1, "s-k", ms=3, lw=1, color=c_25ca)
    axes2[0].plot(range(1, 6), mean_om2[where(wspktr2 > 0)[0]] / munorm1, "s-r", ms=3, lw=1, color=c_13ca)
    axes2[0].set_ylim((-0.5, 10))
    axes2[0].set_yticks((0, 5, 10))
    axes2[0].set_xticklabels([])
    axes2[0].set_ylabel("mean")
    axes2[0].set_xticks(range(1,6))

    axes2[1].plot(range(1, 6), sig1[where(wspktr1 > 0)[0]] / munorm1, "s-k", ms=3, lw=1, color=c_25ca)
    axes2[1].plot(range(1, 6), sig2[where(wspktr2 > 0)[0]] / munorm1, "s-r", ms=3, lw=1, color=c_13ca)
    axes2[1].set_ylim((-0.1, 2))
    axes2[1].set_ylabel("S.D.")
    axes2[1].set_xticklabels([])
    axes2[1].set_xticks(range(1,6))
    axes2[1].set_yticks([0, 1, 2])

    axes2[2].plot(range(1, 6),sig1[where(wspktr1 > 0)[0]] / (mean_om1[where(wspktr1 > 0)[0]]),"s-k",ms=3,lw=1,color=c_25ca,)
    axes2[2].plot(range(1, 6),sig2[where(wspktr2 > 0)[0]] / (mean_om2[where(wspktr2 > 0)[0]]),"s-r",ms=3,lw=1,color=c_13ca,)
    axes2[2].set_ylim((0.1, 0.6))
    axes2[2].set_xlabel("spike nr.")
    axes2[2].set_ylabel("CV")
    axes2[2].set_xticks(range(1,6))
    axes2[2].set_yticks([0.2, 0.4, 0.6])     
    
    return axes2


def plot_histograms(axes3):
    munorm1 = mean_om1[where(wspktr1 > 0)[0][0]]
    mu1,sigma1 = mean_om1[where(wspktr1 > 0)[0][0]], sig1[where(wspktr1 > 0)[0][0]]
    x1 = random.gamma(sigma1, scale=mu1 / sigma1, size=1000)
    
    mu2,sigma2 = mean_om2[where(wspktr2 > 0)[0][0]], sig2[where(wspktr2 > 0)[0][0]]
    x2 = random.gamma(sigma2, scale=mu2 / sigma2, size=1000)
    
    axes3[0].hist(x1 / munorm1,40,range=(0, 5),density=True,color=c_25ca,histtype=u"step",zorder=10,lw=0.75,)
    axes3[0].hist(x2 / munorm1,40,range=(0, 5),density=True,color=c_13ca,histtype=u"step",zorder=0,lw=0.75,)
    
    axes3[0].set_xlabel("norm. PSC")
    axes3[0].set_ylabel("prob. density")
    axes3[0].set_xlim((0, 4))
    #sns.despine()

    mu1,sigma1 = mean_om1[where(wspktr1 > 0)[0][4]], sig1[where(wspktr1 > 0)[0][4]]
    x1 = random.gamma(sigma1, scale=mu1 / sigma1, size=1000)
    
    mu2, sigma2 = mean_om2[where(wspktr2 > 0)[0][4]], sig2[where(wspktr2 > 0)[0][4]]
    x2 = random.gamma(sigma2, scale=mu2 / sigma2, size=1000)
    
    axes3[1].hist(x1 / munorm1,60,range=(0, 20),density=True,color=c_25ca,histtype=u"step",zorder=10,lw=0.75,)
    axes3[1].hist(x2 / munorm1,60,range=(0, 20),density=True,color=c_13ca,histtype=u"step",zorder=20,lw=0.75,)
    
    axes3[1].set_xlim((0, 20))
    axes3[1].set_xlabel("norm. PSC")
    axes3[1].yaxis.set_label_coords(-0.15, 1.0)
    sns.despine()

    return axes3

def plot():
    fig1 = figure(num=1, figsize=figsize)
    gs = GridSpec(6, 3, wspace=0.5, bottom=0.1, hspace=1)
    
    axA1 = fig1.add_subplot(gs[0:2, 0])
    axA2 = fig1.add_subplot(gs[0:2, 1])
    axB = fig1.add_subplot(gs[2:4, 0:2])
    axD = fig1.add_subplot(gs[0:2, 2:])
    axE = fig1.add_subplot(gs[2:4, 2:])
    axF = fig1.add_subplot(gs[4:6, 2:])
    axC1 = fig1.add_subplot(gs[4:6, 0])
    axC2 = fig1.add_subplot(gs[4:6, 1])
    axes1 = [axA1,axA2]
    axes2 = [axD,axE,axF]
    axes3 = [axC1,axC2]
    
    axes1 = plot_kernels(axes1)
    axB = plot_traces(axB)
    axes2 = plot_mu_sig(axes2)
    axes3 = plot_histograms(axes3)
    
    add_figure_letters([axA1, axB, axC1, axD, axE, axF], size=12)
    return fig1

# INITIALIZE PARAMETERS 
# # # # # # # # # #

dt, T = 0.1, 2e3  # ms per bin, in ms
t, Ntrial = arange(0, T, dt), 20
model = "gamma"
nlin = "sigmoid"

a, a2 = 560.0, 600.0 #amplitudes of (mu, sigma) kernels
tauk, mufac = 400.0, 5.0 
b, b2, b3, b4 = -2., -1.0, -6.0, -6.0

wspktr1, I1, mean_om1, sig1, k1 = exec_func(a,a2,b,b2,tauk)
wspktr2, I2, mean_om2, sig2, k2 = exec_func(a,a2,b3,b4,tauk)


if __name__ == "__main__":
    import os, inspect

    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)

    fig = plot()
    plt.savefig(current_dir + "/figures/Fig6_raw.pdf")
    plt.show()

plt.show()
#fig1.savefig("../Figures/Fig6-raw.pdf", format="pdf")


# In[ ]:





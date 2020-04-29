import numpy as np
from scipy import signal
from tools import poisson_simple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from LNLmodel import exponential_kernels, det_exp

plt.style.use('science')
matplotlib.rc('xtick', top=False)
matplotlib.rc('ytick', right=False)
matplotlib.rc('ytick.minor', visible=False)
matplotlib.rc('xtick.minor', visible=False)
plt.rc('font', size=30)

x_ax = np.arange(0, 1000, 0.1)
spktr = poisson_simple(10000, 0.1, 10)
Kpsc = -exponential_kernels(50, 1000, 0.1)[0]

a = 200
tau = 100
b = -2
synapse = det_exp(a, tau, b, T=1000)
efficacy = synapse.run(spktr)['efficacy']

conv = signal.convolve(efficacy, Kpsc, 'full')[:10000]

fig = plt.figure(constrained_layout=True, figsize=(12, 14))
spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[2, 0])
ax4 = fig.add_subplot(spec[3, 0])


ax1.plot(x_ax, spktr, c='black', lw = 3)
ax1.set_title('Spiketrain $S(t)$')
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.axis('off')

ax3.plot(x_ax, np.roll(Kpsc,1), c='black', lw = 3)
ax3.set_title('PSC kernel $\kappa_{PSC}$')
ax3.set_yticks([])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.axis('off')

ax2.plot(x_ax, efficacy, c='red', lw = 3)
ax2.set_title('Efficacy Train $E(t)$')
ax2.set_yticks([])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.axis('off')

ax4.plot(x_ax, conv, c='purple', lw = 3)
ax4.set_title(r'Convolution $\mathbf{\kappa}_{PSC} {\ast} E$')
ax4.set_yticks([])
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.set_xlabel('time (ms)')

plt.savefig('dynamics.png')
plt.show()

# DYNAMICS PRESYNAPTIC

# Plot 1 Spiketrain

# Plot 2 Kernel + Filtered Spiketrain

# Plot 3 Nonlinear Readout

# Plot 4 Efficacy train


#import libraries for easy_fit_SRP
import numpy as np
import math
from srplasticity.srp import ExpSRP
from scipy.optimize import shgo


#import libraries for other fitting and plotting functions
import string
import matplotlib.pyplot as plt
from spiffyplots import MultiPanel
#the functions in the import below are redefined in the code
#from srplasticity.inference import _default_parameter_bounds, _convert_fitting_params

#These functions support new functionality above and beyond the original
#PLOS Computational Biology paper with the primary objective of increasing
#ease of use for the package

#fitting functions
#--------------------------------------------------------------------

def _convert_fitting_params(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1 : 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps : 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )

#--------------------------------------------------------------------

def _default_parameter_bounds(mu_taus, sigma_taus):
    """ 
    returns default parameter boundaries for the SRP fitting procedure
    
    Note 1: Adjusting these is a fruitful area for hyper-parameter tuning
            if you struggle fitting your specific dataset
            
    Note 2: Sigma parameters are only relevant to some versions of the model 
    """
    return [
        (-6, 6),  # mu baseline
        *[(-200, 200), (-1200, 1201), (-5000, 5000)], #timescales for mean dynamics
        (-6, 6),  # sigma baseline
        *[(-10 * tau, 10 * tau) for tau in sigma_taus],  # sigma amps
        (0.001, 100),  # sigma scale
    ]

#--------------------------------------------------------------------

def fit_srp_model(
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    initial_mu_baseline = [0],
    initial_mu=[0.01,0.01,0.01], #default  [0.1,0.1, 0.1]
    initial_sigma_baseline=[-1.8],
    initial_sigma=[0.1,0.1,0.1],
    sigma_scale=[4],
    mu_scale=None,
    bounds="default",
    **kwargs
):
    """
    Fitting the SRP model to data using scipy.optimize.minimize
    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param mu_taus: predefined time constants for mean kernel
    :param sigma_taus: predefined time constants for sigma kernel
    :param initial_*: initialisation value for minization for "*"
    :param mu_scale: mean scale, defaults to None for normalized data
    :param mu_scale: sigma scale, placeholder for compatibility
    :param bounds: bounds for parameters
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize using SHGO
    """

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)  
    
    #select mu params while holding sigmas fixed
    optimizer_res = shgo(
        _objective_function,
        bounds=bounds[0:(len(mu_taus)+2)],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, 
              initial_sigma_baseline, initial_sigma),
        iters=1,
        **kwargs
    )
    
    mse = optimizer_res.fun
    SD = math.pow(mse, 0.5)
    
    params = _convert_fitting_params(
        list(optimizer_res.x)+initial_sigma_baseline+initial_sigma, mu_taus,
        sigma_taus, mu_scale)

    fitted_mu_baseline = params[0]
    fitted_mu_amps = params[1]

    output = (fitted_mu_baseline, fitted_mu_amps, mu_taus, SD, mu_scale)
    
    return output, optimizer_res

#--------------------------------------------------------------------

def mse_loss(target_vals, mean_predicted):
    """
    Stand in Mean Squared error for training loss in first phase
    :param target_vals: (np.array) set of amplitudes
    :param mean_predicted: (np.array) set of means
    """
    loss = []
    for key in target_vals.keys():
        vals_by_amp = [[] for i in range(0, 5)] #2d list for vals by amp
        for i in range(0, len(target_vals[key])):
            run_arr = target_vals[key][i] #get amplitudes from a single run
            run_err = []
            
            if not np.isscalar(run_arr):
                for j in range(0, len(run_arr)):
                    vals_by_amp[j].append(run_arr[j])
                    run_err.append(math.pow((run_arr[j]-mean_predicted[key][j]), 2))
                loss.append(run_err)
    print("loss= "+str(np.nanmean(loss)))
    return np.nanmean(loss)
    
#--------------------------------------------------------------------

def _objective_function(x, *args, phase=0):
    """
    Objective function for scipy.optimize.minimize
    :param x: parameters for SRP model as a list or array:
                [mu_baseline, *mu_amps,
                sigma_baseline, *sigma_amps, sigma_scale]
    :param phase: 0 indicates fitting only mu amps and mu baseline for fixed 
                    sigmas, 1 indicates fitting sigma params for fixed mu params
    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # Unroll arguments
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, fixed_baseline, fixed_amps = args

    new_x = np.append(x, [fixed_baseline]) #add fixed sigma params
    new_x = np.append(new_x, fixed_amps)
    model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus))
        
    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    return _total_loss_det(target_dict, mean_dict)

#--------------------------------------------------------------------

def easy_fit_srp(stimulus_dict, target_dict, mu_kernel_taus=[15, 200, 300], bounds='default'):
    #placeholder sigmas for format, we don't use these for this version
    sigma_kernel_taus = [15, 100, 300]
    
    #generate range of baseline bounds
    best_loss = None
    best_vals = None
    for i in range(-6, 6):
        if bounds == 'default':
            bounds = _default_parameter_bounds(mu_kernel_taus, sigma_kernel_taus)
        bounds[0] = (i, i+1)
        srp_params, optimizer_res = fit_srp_model(stimulus_dict, target_dict, mu_kernel_taus, sigma_kernel_taus, bounds=bounds)
        
        if best_loss == None:
            best_loss = optimizer_res.fun
            best_vals = srp_params
        elif best_loss > optimizer_res.fun:
            best_loss = optimizer_res.fun
            best_vals = srp_params
    return (best_vals, best_loss)

#additional Functions

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plotting
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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


plt.rc("font", family="calibri")


def plot_fit(axis, model, target_dict, stimulus_dict, name_protocol, protocols=None):

    mean, sigma, _ = model.run_ISIvec(stimulus_dict[name_protocol])
    xax = np.arange(1, len(mean) + 1)

    if type(target_dict[name_protocol][0]) is not np.float64:
        errors = np.nanstd(target_dict[name_protocol], 0) / 2

        axis.errorbar(
            xax,
            np.nanmean(target_dict[name_protocol], 0),
            yerr=errors,
            color="black",
            marker="s",
            capsize=2,
            lw=1,
            elinewidth = 0.7,
            markersize=3,
            label="Data",
        )

    axis.text(
        -0.10,
        1.046,
        string.ascii_uppercase[0],
        transform=axis.transAxes,
        size=11,
        weight="bold",
        usetex=False,
        family="calibri",
    )

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.plot(xax, mean, color="#cc3311", label="SRP model")
    if protocols != None:
        axis.set_title(protocols[name_protocol], fontweight='semibold', )
    else:
        axis.set_title(name_protocol)
    axis.set_xticks(xax)
    axis.set_ylim(0.5, 9)
    axis.set_yticks([1, 3, 5, 7, 9])

    axis.legend(frameon=False)
    axis.set_ylabel("norm. EPSC")
    axis.set_xlabel("spike nr.")

def plot_mse_fig(axis, mses):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.boxplot(mses, medianprops = dict(color="#03719c", linewidth=1.25), showfliers=False)
    axis.set_ylabel("MSE", labelpad=8)
    axis.set_ylim(-0.05, 0.3)
    axis.set_yticks([0, 0.1, 0.2, 0.3])
    axis.set_xticks([])

    axis.text(
        -0.15,
        1.1,
        string.ascii_uppercase[1],
        transform=axis.transAxes,
        size=11,
        weight="bold",
        usetex=False,
        family="calibri",
    )


def gen_kernel(mu_amps, mu_taus, mu_baseline=None, dt=1):
    # set up timespan of 2000ms with 0.1ms time bins
    dt = 0.1  # ms per bin
    T = 2e3  # in ms

    t = np.arange(0, T, dt)
    spktr = np.zeros(t.shape)
    spktr[[4000]] = 1 #set one spike at 400ms

    kernels = [1 / tauk * np.exp(-t / tauk) for tauk in mu_taus] #generate mu kernels wrt time

    if mu_baseline == None:
        mu_baseline = 0

    # y_vals = np.roll(mu_amps[0] * kernels[0][:10000] + mu_amps[1] * kernels[1][:10000]+ mu_amps[2] * kernels[2][:10000] + mu_baseline, 2000)
    y_vals = np.roll(sum([mu_amps[i] * kernels[i][:10000] for i in range(len(mu_taus))]) + mu_baseline, 2000)
    x_vals = t[:10000] - 200
    #set pre-spike baseline, figure out why this isn't always the case
    for i in range(0, 2000):
        y_vals[i] = mu_baseline
    return (x_vals, y_vals)


def plot_kernel(axis, mu_taus, mu_amps, mu_baseline, colour="#03719c"):
    # #047495
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_taus, mu_baseline=mu_baseline)
    axis.plot(kernel_x, kernel_y, color=colour)
    axis.set_ylim(-2.5, -0.5)
    axis.set_yticks([-2.5, -1.5, -0.5])
    axis.set_ylabel("Efficacy kernel", labelpad=1)

    axis.text(
        -0.15,
        1.1,
        string.ascii_uppercase[2],
        transform=axis.transAxes,
        size=11,
        weight="bold",
        usetex=False,
        family="calibri",
    )


def plot_fig(model, mu_baseline, mu_amps, mu_taus, target_dict, stimulus_dict, mses, chosen_protocol, protocol_names):
    # fig = MultiPanel(grid=[(0, range(2)), (0, 2), (0, 3)], figsize=(9.5, 3.25))
    fig = MultiPanel(grid=[(range(2), range(2)), (0, 2), (1, 2)], figsize=(7, 3.75), wspace=0.5, hspace=0.2)

    plot_fit(fig.panels[0], model, target_dict, stimulus_dict, chosen_protocol, protocols=protocol_names)

    plot_mse_fig(fig.panels[1], mses)

    plot_kernel(fig.panels[2], mu_taus, mu_amps, mu_baseline)

    # add_figure_letters([fig.panels[ix] for ix in axes_with_letter], 11)
    plt.show()


# Plot mean fit
def plot_srp(model, target_dict, stimulus_dict, protocols=None):

    npanels = len(list(target_dict.keys()))

    if npanels > 1:
        fig = MultiPanel(grid=[npanels], figsize=(npanels * 3, 3))

        for ix, key in enumerate(list(target_dict.keys())):
            mean, sigma, _ = model.run_ISIvec(stimulus_dict[key])
            xax = np.arange(1, len(mean) + 1)

            if type(target_dict[key][0]) is not np.float64:
                errors = np.nanstd(target_dict[key], 0) / 2

                fig.panels[ix].errorbar(
                    xax,
                    np.nanmean(target_dict[key], 0),
                    yerr=errors,
                    color="black",
                    marker="o",
                    markersize=2,
                    label="Data",
                )
            fig.panels[ix].plot(xax, mean, color="#cc3311", label="SRP model")
            if protocols != None:
                fig.panels[ix].set_title(protocols[key])
            else:
                fig.panels[ix].set_title(key)
            fig.panels[ix].set_xticks(xax)
            fig.panels[ix].set_ylim(0.5, 9)
            fig.panels[ix].set_yticks([1, 3, 5, 7, 9])

        fig.panels[0].legend(frameon=False)
        fig.panels[0].set_ylabel("norm. EPSC amplitude")

        plt.show()


def plot_estimates(means):
    fig, axis = plt.subplots()
    xax = range(len(means))
    axis.plot(range(1, len(xax) + 1), means, lw=1, color="#cc3311")
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_ylabel("Predicted EPSC", labelpad=1, size=8)
    axis.set_xticks(range(1, len(xax) + 1))
    axis.set_xlabel("spike nr.", labelpad=1)
    fig.set_dpi(1200)
    fig.tight_layout()
    # plt.savefig(f"estimates_plot.svg", transparent=True)


def plot_spike_train(spiketrain):
    fig, axis = plt.subplots()
    axis.plot(spiketrain, lw=0.7, color='black')
    axis.set_ylim(-0.005, 0.005)
    axis.axis("off")

    fig.set_dpi(1200)
    fig.tight_layout()
    # plt.savefig(f"spike_train_plot.svg", transparent=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# FITTING FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def fit_srp_model_fssd(
        stimulus_dict,
        target_dict,
        mu_taus,
        mu_scale=None,
        bounds="default",
        loss="default",
        **kwargs
):
    sigma_taus = mu_taus
    initial_sigma_baseline = [-1.8]
    initial_sigma = [0.1]

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)
    initial_sigma = initial_sigma * len(mu_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)

    optimizer_res = shgo(
        _objective_function_det,
        bounds=bounds[0:len(mu_taus) + 1],
        args=(
        target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma),
        iters=1,
        **kwargs
    )

    mse = optimizer_res.fun
    SD = math.pow(mse, 0.5)

    params = _convert_fitting_params(list(optimizer_res.x) + initial_sigma_baseline + initial_sigma, mu_taus,
                                     sigma_taus, mu_scale)

    fitted_mu_baseline = params[0]
    fitted_mu_amps = params[1]

    output = (fitted_mu_baseline, fitted_mu_amps, mu_taus, SD, mu_scale)

    return output, params, optimizer_res

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS FOR FITTING PROCEDURE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _total_loss_det(target_vals, mean_predicted):

    """
    Stand in Mean Squared error for training loss in first phase
    :param target_vals: (np.array) set of amplitudes
    :param mean_predicted: (np.array) set of means
    """

    loss = []

    for key in target_vals.keys():
        for i in range(0, len(target_vals[key])):
            run_arr = target_vals[key][i]  # get amplitudes from a single run
            run_err = []

            if not np.isscalar(run_arr):
                for j in range(0, len(run_arr)):
                    run_err.append(math.pow((run_arr[j] - mean_predicted[key][j]), 2))
                loss.append(run_err)
    loss_2 = []
    for i in loss:
        for j in i:
            loss_2.append(j)

    total_mse_loss = np.nanmean(loss_2)

    return total_mse_loss


def _objective_function_det(x, *args):
    """
    Objective function for scipy.optimize.minimize

    :param x: parameters for SRP model as a list or array:
                [mu_baseline, *mu_amps,
                sigma_baseline, *sigma_amps, sigma_scale]

    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # Unroll arguments
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma = args

    new_x = np.append(x, [initial_sigma_baseline])
    new_x = np.append(new_x, initial_sigma)

    # Initialize model
    model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus, mu_scale))

    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    return _total_loss_det(target_dict, mean_dict)



def get_poisson_ISIs(nspikes, rate):
    """
    poisson ISIs
    :param nspikes: number of spikes
    :param rate: firing rate in Hz
    """

    meanISI = int(1000 / rate)
    isis = np.random.exponential(scale=meanISI, size=nspikes).round(1)
    isis[isis < 2] = 2  # minimum 2ms refractory period
    return isis


def compute_mses(stimulus_dict, target_dict, model):
    mses = []
    for protocol in stimulus_dict.keys():
        means, _, _ = model.run_ISIvec(stimulus_dict[protocol])
        for j in range(len(means)):
            mses.append(np.square(np.nanmean(target_dict[protocol][:, j]) - means[j]))
    return mses

#import libraries for easy_fit_SRP
import numpy as np
import math
from srplasticity.srp import easySRP
from scipy.optimize import shgo

#import libraries for other fitting and plotting functions
import string
import matplotlib.pyplot as plt
from spiffyplots import MultiPanel
#the functions in the import below are redefined in the code

#These functions support new functionality above and beyond the original
#PLOS Computational Biology paper with the primary objective of increasing
#ease of use for the package

#--------------------------------------------------------------------

def _EasySRP_dict_to_tuple(in_dict):
    """
    Convert dict of easySRP parameters to a standardized tuple
    
    :param in_dict: parameter dict 
    :type: dict
    
    :return: tuple of easySRP parameters
    """

    return (in_dict["mu_baseline"], in_dict["mu_amps"], in_dict["mu_taus"], 
            in_dict["SD"], in_dict["mu_scale"])

#--------------------------------------------------------------------

def _EasySRP_tuple_to_dict(in_tuple):
    """
    Convert standardized tuple of easySRP parameters to a dict
    
    :param in_tuple: standardized tuple 
    :type: tuple
    
    :return: Dict of easySRP parameters
    """
    return {"mu_baseline":in_tuple[0],
            "mu_amps":in_tuple[1],
            "mu_taus":in_tuple[2],
            "SD":in_tuple[3],
            "mu_scale":in_tuple[4]}

#--------------------------------------------------------------------

def _default_parameter_bounds():
    """ 
    returns default parameter boundaries for the SRP fitting procedure
    These bounds assume mu_taus = [15ms, 200ms, 300ms]
    
    :return: list of tuples containining easySRP parameter bonunds
    """
    return [(-6, 6),  # mu baseline
            #timescales for mean dynamics
            *[(-200, 200), (-1200, 1201), (-5000, 5000)]]

#--------------------------------------------------------------------

def fit_srp_model(
    stimulus_dict,
    target_dict,
    mu_taus,
    mu_scale=None,
    bounds="default",
    **kwargs
):
    """
    Fitting the SRP model to data using scipy.optimize.minimize
    
    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :type: dict
    :param target_dict: mapping of protocol keys to response matrices
    :type: dict
    :param mu_taus: predefined time constants for mean kernel
    :type: list of ints
    :param mu_scale: mean scale, defaults to None for normalized data
    :type: float
    :param bounds: bounds for parameters
    :type: list of tuples (min_value, max_value)
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    
    :return: output of scipy.minimize using SHGO
    """

    mu_taus = np.atleast_1d(mu_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds()  
    
    #select mu params while holding sigmas fixed
    optimizer_res = shgo(
        _objective_function,
        bounds=bounds[0:(len(mu_taus)+1)],
        args=(target_dict, stimulus_dict, mu_taus, mu_scale),
        iters=1,
        **kwargs
    )
    
    mse = optimizer_res.fun
    SD = math.pow(mse, 0.5)
    fitted_mu_baseline = optimizer_res.x[0]
    fitted_mu_amps = optimizer_res.x[1:len(mu_taus)+1]

    easySRP_params = _EasySRP_tuple_to_dict((fitted_mu_baseline, fitted_mu_amps,
                                            mu_taus, SD, mu_scale))
    
    return easySRP_params, optimizer_res

#--------------------------------------------------------------------

def mse_loss(target_vals, mean_predicted):
    """
    Mean Squared error for training loss
    
    :param target_vals: Dict of dict numpy arrays containing sets of amplitudes
                        outer dict has keys for synapse inner dict has keys for
                        protocol
    :type target_vals: dict
    :param mean_predicted: Model predict response means
    :type mean_predicted: Numpy array
    
    :return: mean squared error 
    """
    loss = []
    for protocol, responses in target_vals.items():
        loss.append(np.square(responses-mean_predicted[protocol]).flatten())
    final_loss = np.nanmean(np.concatenate(loss))
    print("loss = "+str(final_loss))
    return final_loss
    
#--------------------------------------------------------------------

def _objective_function(x, *args):
    """
    Objective function for scipy.optimize.minimize
    
    :param x: parameters for SRP model: [mu_baseline, *mu_amps]
    :type x: 1-D array
    :param args: target dictionary (dict of responses by protocol) 
                    and stimulus dictionary (dict of stimuli by protocol)
    :type args: tuple
                
    :return: total loss to be minimized
    """
    # Unroll arguments
    target_dict, stimulus_dict, mu_taus, mu_scale = args
    
    mu_baseline = x[0]
    mu_amps = x[1:len(mu_taus)+1]
    
    model_params = {"mu_baseline":mu_baseline,
                    "mu_amps":mu_amps,
                    "mu_taus":mu_taus}
    
    model = easySRP(**model_params)
        
    # compute estimates
    mean_dict = {}

    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], efficacies = model.run_ISIvec(ISIvec)

    return mse_loss(target_dict, mean_dict)

#--------------------------------------------------------------------

def easy_fit_srp(stimulus_dict, target_dict, mu_kernel_taus=[15, 200, 300],
                 bounds='default'):
    """
    Introductory function to fit an SRP model with fixed Gaussian variance
    and history dependent mean behaviour with the best "out of the box" 
    performance by running multiple fits of constrained ranges of the 
    mu_baseline parameter.
    
    :param stimulus_dict: Dict of different stimulus protocols
    :type stimulus_dict: dict of numpy arrays
    :param target_dict: Dict of numpy arrays containing observed responses
                            with keys corresponding to different protocols 
    :type target_dict:dict of numpy arrays
    :param mu_kernel_taus: List of taus for exponential decays that make up
                            mu_kernel
    :type mu_kernel_taus: list of int
    
    :return: Tuple containing: Dict of best easySRP fitted model parameters,
            corresponding fit loss
    """
    
    #generate range of baseline bounds
    best_loss = None
    best_vals = None
    for i in range(-6, 6):
        if bounds == 'default':
            bounds = _default_parameter_bounds()
        bounds[0] = (i, i+1)
        srp_params, optimizer_res = fit_srp_model(stimulus_dict, target_dict, 
                                                  mu_kernel_taus, bounds=bounds)
        
        if best_loss == None:
            best_loss = optimizer_res.fun
            best_vals = srp_params
        elif best_loss > optimizer_res.fun:
            best_loss = optimizer_res.fun
            best_vals = srp_params
    return (best_vals, best_loss)


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
    
    #this section is unneccessary and confusing: numpy already flattens the array
    loss_2 = []
    for i in loss:
        for j in i:
            loss_2.append(j)

    total_mse_loss = np.nanmean(loss_2)

    return total_mse_loss


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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# OTHER
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def norm_responses(target_dict):
    first_spike_list = []
    normed_all = {}
    for protocol in target_dict.keys():
        try:
            divisors = target_dict[protocol][:, 0]
            for i in range(0, len(divisors)):
                first_spike_list.append(divisors[i])
        except:
            print("no entry")

    if len(first_spike_list) > 0:
        averaged_divisor = np.nansum(first_spike_list) / len(first_spike_list)
        print(f"Averaged divisor: {averaged_divisor}")

        for protocol in target_dict.keys():
            normed_all[protocol] = []
            for i in range(0, len(target_dict[protocol])):
                normed_row = target_dict[protocol][i]
                normed_row = normed_row / averaged_divisor
                if len(normed_all[protocol]) == 0:
                    normed_all[protocol] = normed_row
                else:
                    normed_all[protocol] = np.vstack([normed_all[protocol], normed_row])
    return normed_all

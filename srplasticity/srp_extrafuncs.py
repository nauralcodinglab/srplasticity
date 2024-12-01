#import libraries for easy_fit_SRP
import numpy as np
import math
from srplasticity.srp import (easySRP, ExpSRP)
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
    Function to add Letters enumerating multipanel figures

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


def plot_fit(axis, model, target_dict, stimulus_dict, name_protocol, protocols=None):
    """
    Plot the fit of the model to the target data for a given protocol

    :param axis: The axis on which to plot the data
    :type axis: matplotlib.axes.Axes
    :param model: The SRP model with history dependent mean behaviour and fixed variance
    :type model: class: 'easySRP'
    :param target_dict: Dictionary where keys are protocol names 
                        and values are NumPy arrays of the responses
    :type target_dict: dict
    :param stimulus_dict: Dictionary where keys are protocol names 
                            and values are lists of ISIs
    :type stimulus_dict: dict
    :param name_protocol: The name of the protocol to be plotted
    :type name_protocol: str
    :param protocols: Dictionary where keys are protocol names (str) 
                      and values are their descriptive names (str). 
                      Defaults to None
    :type protocols: dict, optional
    """

    mean = model.run_ISIvec(stimulus_dict[name_protocol])[0]
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

    plt.show()

def plot_mse_fig(axis, mses):
    """
    Plot the Mean Squared Error (MSE) as a boxplot on the given axis

    :param axis: The axis on which to plot the MSE boxplot
    :type axis: matplotlib.axes.Axes
    :param mses: A list or array of MSE values to be plotted
    :type mses: list
    """

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.boxplot(mses, medianprops = dict(color="#03719c", linewidth=1.25), showfliers=False)
    axis.set_ylabel("MSE", labelpad=8)
    axis.set_xticks([])

    axis.text(
        -0.15,
        1.1,
        string.ascii_uppercase[1],
        transform=axis.transAxes,
        size=11,
        weight="bold",
        usetex=False,
    )

    plt.show()


def _arrange_kernel(mu_amps, mu_taus, mu_baseline=None, dt=0.1):
    """
    Generate and arrage a kernel to be plotted by the "plot_kernel" function.
    This function sets up a timespan of 2000 ms with 0.1 ms time bins, generate a kernel based on the provided
    amplitudes and time constants, and returns the time values and corresponding kernel values.

    :param mu_amps: List of amplitudes for the kernels
    :type mu_amps: list
    :param mu_taus: List of time constants for the kernels
    :type mu_taus: list
    :param mu_baseline: Baseline value to be added to the kernel. Defaults to None
    :type mu_baseline: float, optional
    :param dt: Time step for the kernel generation. Defaults to 1 ms
    :type dt: float, optional

    :return: A tuple containing:
                - x_vals (np.ndarray): Array of time values
                - y_vals (np.ndarray): Array of kernel values
    """

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


def plot_kernel(axis, taus, amps, baseline, colour="#03719c"):
    """
    Plot the efficacy kernel on the given axis.

    :param axis: The axis on which to plot the kernel
    :type axis: matplotlib.axes.Axes
    :param taus: The time constants of the exponentials that compose the kernel
    :type taus: numpy array
    :param amps: The amplitudes of the exponentials that compose the kernel
    :type amps: numpy array
    :param baseline: The baseline parameter of the kernel
    :type baseline: float
    :param colour: Colour of the kernel plot. Defaults to #03719c
    :type colour: str, optional
    """

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    kernel_x, kernel_y = _arrange_kernel(amps, taus, mu_baseline=baseline)
    axis.plot(kernel_x, kernel_y, color=colour)
    axis.set_ylim(-2.5, -0.5)
    axis.set_yticks([-2.5, -1.5, -0.5])
    axis.set_ylabel("Efficacy kernel", labelpad=1)
    axis.set_xlabel("Time (ms)", labelpad=1)

    axis.text(
        -0.15,
        1.1,
        string.ascii_uppercase[2],
        transform=axis.transAxes,
        size=11,
        weight="bold",
        usetex=False,
    )

    plt.show()


def plot_fig(params, target_dict, stimulus_dict, mses, chosen_protocol, protocol_names=None):
    """
    Generate a multi-panel figure to visualize model fit, MSE, and efficacy kernel.

    This function creates a multi-panel figure with three subplots:
    1. The fit of the model to the target data for the chosen protocol.
    2. A boxplot of the MSE values.
    3. The efficacy kernel based on the provided amplitudes, time constants, and baseline value.

    :param params: Dict of easySRP parameters:
                   {"mu_baseline": float,
                    "mu_amps": numpy array,
                    "mu_taus": numpy array,
                    "SD": float,
                    "mu_scale": int, float or None}
    :type params: dict
    :param target_dict: Dictionary where keys are protocol names 
                        and values are NumPy arrays of the responses
    :type target_dict: dict
    :param stimulus_dict: Dictionary where keys are protocol names 
                        and values are lists of ISIs
    :type stimulus_dict: dict
    :param mses: List or array of Mean Squared Error (MSE) values
    :type mses: list
    :param chosen_protocol: The name of the protocol to be plotted
    :type chosen_protocol: str
    :param protocol_names: Dictionary where keys are protocol names (str)
                           and values are their descriptive names (str). 
                           Defaults to None
    :type protocol_names: dict, optional
    """


    try:
        mu_baseline = params["mu_baseline"]
        mu_amps = params["mu_amps"]
        mu_taus = params["mu_taus"]

        model = easySRP(**params)
    except (TypeError, KeyError):
        raise ValueError("'params' must correspond to a dict of easySRP parameters")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

    fig = MultiPanel(grid=[(range(2), range(2)), (0, 2), (1, 2)], figsize=(7, 3.75), wspace=0.5, hspace=0.2)

    plot_fit(fig.panels[0], model, target_dict, stimulus_dict, chosen_protocol, protocols=protocol_names)

    plot_mse_fig(fig.panels[1], mses)

    plot_kernel(fig.panels[2], mu_taus, mu_amps, mu_baseline)

    # add_figure_letters([fig.panels[ix] for ix in axes_with_letter], 11)
    plt.show()


def plot_srp(params, target_dict, stimulus_dict, protocols=None, srp_model='easySRP'):
    """
    Plot the Spike Response Plasticity (SRP) model fit for multiple protocols

    :param params: Dict of easySRP parameters:
                    {"mu_baseline": float,
                     "mu_amps": numpy array,
                     "mu_taus": numpy array,
                     "SD": float,
                     "mu_scale": int, float or None}
    :type params: dict
    :param target_dict: Dictionary where keys are protocol names 
                        and values are NumPy arrays of the responses
    :type target_dict: dict
    :param stimulus_dict: Dictionary where keys are protocol names 
                          and values are lists of ISIs
    :type stimulus_dict: dict
    :param protocols: Dictionary where keys are protocol names (str)
                      and values are their descriptive names (str). 
                      Defaults to None
    :type protocols: dict, optional
    :param srp_model: Name of the model class to be instantiated. 
                      Defaults to 'easySRP'
    :type srp_model: str, optional
    """
    if srp_model != 'easySRP' and srp_model != 'ExpSRP':
      raise ValueError("srp_model must be easySRP or ExpSRP")
    
    try:
      if srp_model == 'easySRP':
        model = easySRP(**params)
        params["SD"]
      else:
        model = ExpSRP(**params)
        params["sigma_baseline"]
    except (TypeError, KeyError):
        raise ValueError("'params' must be a dictionary of either easySRP or ExpSRP parameters, but not both. "
                        "Ensure that the provided 'params' dictionary aligns with the selected 'srp_model'.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

    npanels = len(list(target_dict.keys()))

    if npanels > 1:
        fig = MultiPanel(grid=[npanels], figsize=(npanels * 3, 3))

        for ix, key in enumerate(list(target_dict.keys())):
          if srp_model == 'easySRP':
            mean, efficacies = model.run_ISIvec(stimulus_dict[key])
            lower_SD = mean - params["SD"]
            upper_SD = mean + params["SD"]
          else:
            mean, sigma, efficacies = model.run_ISIvec(stimulus_dict[key])
            lower_SD = mean - sigma
            upper_SD = mean + sigma
          
          xax = np.arange(1, len(mean) + 1)

          if type(target_dict[key][0]) is not np.float64:
              errors = np.nanstd(target_dict[key], 0)

              fig.panels[ix].errorbar(
                  xax,
                  np.nanmean(target_dict[key], 0),
                  yerr=errors,
                  color="black",
                  marker="o",
                  markersize=2,
                  label="Data",
                  capsize=2,
              )
          fig.panels[ix].plot(xax, mean, color="#cc3311", label="SRP model")
          fig.panels[ix].fill_between(xax, lower_SD, upper_SD, color="xkcd:light grey", label="SD")
          if protocols != None:
              fig.panels[ix].set_title(protocols[key])
          else:
              fig.panels[ix].set_title(key)
          if len(xax) <= 10:
              fig.panels[ix].set_xticks(xax)
          else:
              ticks = np.arange(1, len(mean) + 1, math.ceil(len(mean) / 10))
              fig.panels[ix].set_xticks(ticks)
          fig.panels[ix].set_ylim(0.5, 9)
          fig.panels[ix].set_yticks([1, 3, 5, 7, 9, 11])

        fig.panels[0].legend(frameon=False)
        fig.panels[0].set_ylabel("norm. EPSC amplitude")

        plt.show()


def plot_estimates(means, efficacies):
    """
    Plot sample estimates over time with the corresponding mean values.

    :param means: A list of mean predicted responses
    :type means: list
    :param efficacies: A list of sample predicted responses
    :type efficacies: list
    """

    fig, axis = plt.subplots()

    if type(efficacies[0]) != np.float64 and type(efficacies[0]) != float:
        xax = range(len(efficacies[0]))

        for i in range(len(efficacies)):
            if i == 0:
                axis.plot(range(1, len(xax) + 1), efficacies[i], lw=1, color="#cc3311", label="Samples")
            else:
                axis.plot(range(1, len(xax) + 1), efficacies[i], lw=1, color="#cc3311")

        axis.plot(range(1, len(xax) + 1), means, lw=1.2, color="black", label='Means')

    else:
        xax = range(len(efficacies))
        axis.plot(range(1, len(xax) + 1), efficacies, lw=1, color="#cc3311", label="Samples")
        axis.plot(range(1, len(xax) + 1), means, lw=1.2, color="black", label='Means')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_ylabel("Predicted EPSC", labelpad=1, size=8)
    axis.set_xticks(range(1, len(xax) + 1))
    axis.set_xlabel("spike nr.", labelpad=1)
    fig.legend(frameon=False)

    plt.show()


def plot_spike_train(spiketrain):
    """
    Plot a spike train

    :param spiketrain: A binary stimulation vector from a
                        vector with ISI intervals to be plotted
    :type spiketrain: Numpy array
    """

    fig, axis = plt.subplots()
    axis.plot(spiketrain, lw=0.7, color='black')
    axis.set_ylim(-1e-5, 5e-6)
    axis.axis("off")

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_poisson_ISIs(nspikes, rate):
    """
    Poisson ISIs

    :param nspikes: number of spikes
    :type nspikes: int
    :param rate: firing rate in Hz
    :type rate: int
    """

    meanISI = int(1000 / rate)
    isis = np.random.exponential(scale=meanISI, size=nspikes).round(1)
    isis[isis < 2] = 2  # minimum 2ms refractory period
    return isis


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PREPROCESSING
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def norm_responses(target_dict):
    """
    Normalizes the responses in the `target_dict` by averaging the first responses values,
    then dividing all responses by this average value.

    :param target_dict: Dictionary where keys are protocol names 
                        and values are NumPy arrays of the responses
    :type target_dict: dict

    :return: A dictionary in which keys are protocol names 
             and values are NumPy arrays of the normalized responses
    """
    first_spike_list = []
    normed_all = {}
    for protocol in target_dict.keys():
        try:
            divisors = target_dict[protocol][:, 0]
            first_spike_list.extend(divisors)
        except:
            print("no entry")

    if len(first_spike_list) > 0:
        averaged_divisor = np.nansum(first_spike_list) / len(first_spike_list)
        print(f"Averaged divisor: {averaged_divisor}")

        for protocol, data in target_dict.items(): 
          normed_all[protocol] = data / averaged_divisor
            
    return normed_all

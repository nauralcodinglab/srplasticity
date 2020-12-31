from abc import ABC, abstractmethod
import numpy as np


class AbstractSynapse(ABC):
    """
    Abstract base method for a synapse model.
    """

    def __init__(self, n, w=1) -> None:
        """
        Constructor for abstract base class for synapse objects.
        :param n: number of synapses
        :param w: weights
        """
        self.n = n

        if w is None:
            self.w = random_weights(self.n)

        elif np.size(w) == 1:  # if w is a single number, scale all synapses by w
            self.w = np.full(self.n, w)

        self.out = np.zeros(self.n)  # empty matrix of synaptic output

    @abstractmethod
    def update(self, s, dt) -> None:
        """
        Update synapse given a vector of presynaptic spikes
        """
        pass

    @abstractmethod
    def reset_(self) -> None:
        """
        Contains resetting logic for the synapse.
        """
        self.out = np.zeros(self.n)  # empty matrix of synaptic output


class StaticSynapse(AbstractSynapse):
    """
    Specifies a static synapse between one or two populations of neurons.
    Upon a spike, a postsynaptic response is evoked with a fixed strength w.
    """

    def __init__(self, n, w=1):
        """
        :param n: number of synapses
        :param w: weight
        """
        super().__init__(n, w)
        super().__init__(n, w)

    def update(self, s, dt) -> None:
        """
        Update synapses given pre-synaptic spikes and a timestep
        :param s: presynaptic spikes (boolean)
        :param dt: timestep
        """

        # Spike vector to matrix considering synaptic strength
        self.out = s * self.w
        super().update(s, dt)

    def run(self, spiketrain, dt) -> dict:
        """
        Computes synaptic output for a given input spike train.
        """

        assert (
            spiketrain.shape[0] == self.n
        ), "Spike train does not correspond to number of synapses."

        # Time vector
        t = np.arange(0, spiketrain.shape[1] * dt, dt)
        assert len(t) == spiketrain.shape[1]
        steps = len(t)

        result = {"psr": np.zeros((self.n, steps)), "t": t}

        for step in range(steps):
            self.update(s=spiketrain[step], dt=dt)
            result["psr"][..., step] = self.out

        self.reset_()

        return result

    def run_ISIs(self, ISIvector):
        """
        Computes synaptic output given a vector of presynaptic ISIs.
        """

        ISIvector[0] = 0  # first spike is always assumed to have no spike history.

    def reset_(self) -> None:
        """
        Resets synapses
        """
        super().reset_()


class gTMSynapse(AbstractSynapse):

    """
    Implements a synapse according to a generalized Tsodyks-Markram model
    """

    def __init__(self, n, U, f, tau_U, Z, tau_Z, tau_R, w=1):
        """
        :param n: number of synapses
        :param w: weight

        :param U: baseline of utilized efficacy u
        :param f: facilitation constant
        :param tau_U: time constant of utilized efficacy
        :param Z: baseline of second facilitation parameter
        :param tau_Z: baseline of second facilitation parameter
        :param tau_R: time constant of available efficacy
        """

        super().__init__(n, w)

        # Constants
        self.U = U
        self.f = f
        self.tau_U = tau_U
        self.Z = Z
        self.tau_Z = tau_Z
        self.tau_R = tau_R

        # Running parameters
        self.u = self.U.copy()
        self.z = self.Z.copy()
        self.R = np.ones(self.n)
        self.deltaS = np.zeros(self.n)  # Time since previous spike
        self.spikecount = np.zeros(self.n)  # Spike counter

    def update(self, s, dt) -> None:
        """
        Update synapses given pre-synaptic spikes and a timestep
        :param s: presynaptic spikes (boolean)
        :param dt: timestep
        """

        # Spike vector to matrix considering synaptic strength

        for n in np.argwhere(s == 1):  # loop through spiking neurons
            if (
                self.spikecount[n] != 0
            ):  # Only update if neuron is not spiking for the first time

                decayR = np.exp(-self.deltaS[n] / self.tau_R[n])
                decayU = np.exp(-self.deltaS[n] / self.tau_U[n])
                decayZ = np.exp(-self.deltaS[n] / self.tau_Z[n])

                self.R[n] = (
                    1
                    - (1 - self.R[n]) * decayR
                    - self.R[n] * self.u[n] * self.z[n] * decayR
                )
                u_jump = self.u[n] + self.f[n] * self.u[n] * (1 - self.u[n])
                self.u[n] = self.U[n] + (u_jump - self.U[n]) * decayU
                self.z[n] = (
                    self.Z[n]
                    + (
                        self.z[n]
                        + np.heaviside(u_jump - self.z[n], 0) * (u_jump - self.z[n])
                        - self.Z[n]
                    )
                    * decayZ
                )

            self.deltaS[n] = 0  # set time since spike to zero
            self.spikecount[n] += 1  # add to spike counter

        # output
        self.out = self.u * self.z * self.R * s * self.w

        # increase time since spike counter
        self.deltaS += dt
        super().update(s, dt)

    def update_all(self, s, dt) -> None:
        """
        Update all synapses, irrespective of whether they spike or not.
        :param s: presynaptic spikes (boolean)
        :param dt: timestep
        """

        # pre-calculate decays
        decayR = np.exp(-self.deltaS / self.tau_R)
        decayU = np.exp(-self.deltaS / self.tau_U)
        decayZ = np.exp(-self.deltaS / self.tau_Z)

        # update synaptic mechanism of all neurons
        self.R_temp = 1 - (1 - self.R) * decayR - self.R * self.u * self.z * decayR
        u_jump = self.u + self.f * self.u * (1 - self.u)
        self.u_temp = self.U + (u_jump - self.U) * decayU
        self.z_temp = (
            self.Z
            + (self.z + np.heaviside(u_jump - self.z, 0) * (u_jump - self.z) - self.Z)
            * decayZ
        )

        spiking = np.argwhere(s == 1).flatten()  # spiking neurons
        againspiking = spiking[
            np.argwhere(self.spikecount[spiking] > 0).flatten()
        ]  # neurons that don't spike for the first time
        neverspiking = np.argwhere(
            self.spikecount == 0
        )  # neurons that don't spike for the first time

        # accept update for neurons that spike for the second or more time
        self.R[againspiking] = self.R_temp[againspiking]
        self.u[againspiking] = self.u_temp[againspiking]
        self.z[againspiking] = self.z_temp[againspiking]

        # Reset update for neurons that have never spiked
        self.R_temp[neverspiking] = 1
        self.u_temp[neverspiking] = self.U[neverspiking]
        self.z_temp[neverspiking] = self.Z[neverspiking]

        # record spike for all spiking neurons
        self.deltaS[spiking] = 0  # set time since spike to zero
        self.spikecount[spiking] += 1  # add to spike counter

        # output
        self.out = self.u * self.z * self.R * s * self.w

        # increase time since spike counter
        self.deltaS += dt
        super().update(s, dt)

    def run(self, spiketrain, dt, update_all=False) -> dict:
        """
        Computes synaptic output for a given input spike train.
        :param spiketrain: np.array of presynaptic spike trains at each time point
        :param dt: timestep
        :update_all: Boolean. If true, every synapse is updated at each time point and the mechanistic parameters
        (u, R, Z) are returned. If false, ODEs are integrated between spikes for efficient numerical implementation.
        """

        assert (
            spiketrain.shape[0] == self.n
        ), "Spike train does not correspond to number of synapses."

        # Time vector
        t = np.arange(0, spiketrain.shape[1] * dt, dt)
        assert len(t) == spiketrain.shape[1]
        steps = len(t)

        if update_all == True:

            self.u_temp = self.u.copy()
            self.R_temp = self.R.copy()
            self.z_temp = self.z.copy()

            # Initialize result matrices
            result = {
                "psr": np.zeros((self.n, steps)),
                "u": np.zeros((self.n, steps)),
                "z": np.zeros((self.n, steps)),
                "R": np.zeros((self.n, steps)),
                "t": t,
            }

            # Loop over spike train
            for step in range(steps):
                self.update_all(s=spiketrain[..., step], dt=dt)
                result["psr"][..., step] = self.out
                result["u"][..., step] = self.u_temp
                result["z"][..., step] = self.z_temp
                result["R"][..., step] = self.R_temp

        else:
            # Initialize result matrices
            result = {"psr": np.zeros((self.n, steps)), "t": t}

            # Loop over spike train
            for step in range(steps):
                self.update(s=spiketrain[..., step], dt=dt)
                result["psr"][..., step] = self.out

        self.reset_()

        return result

    def reset_(self) -> None:
        """
        Resets synapses
        """
        self.u = self.U.copy()
        self.z = self.Z.copy()
        self.R = np.ones(self.n)
        self.deltaS = np.zeros(self.n)  # Time since previous spike
        self.spikecount = np.zeros(self.n)  # Spike counter

        super().reset_()


class cTMSynapse(AbstractSynapse):

    """
    Implements a synapse according to cTM model of dynamic synapses (Tsodyks & Markram, 1997).
    """

    def __init__(self, n, U, f, tau_U, tau_R, w=1):
        """
        :param n: number of synapses
        :param w: weight

        :param U: baseline of utilized efficacy u
        :param f: facilitation constant
        :param tau_U: time constant of utilized efficacy
        :param Z: baseline of second facilitation parameter
        :param tau_Z: baseline of second facilitation parameter
        :param tau_R: time constant of available efficacy
        """

        super().__init__(n, w)

        # Constants
        self.U = U
        self.f = f
        self.tau_U = tau_U
        self.tau_R = tau_R

        # Running parameters
        self.u = self.U.copy()
        self.R = np.ones(self.n)
        self.deltaS = np.zeros(self.n)  # Time since previous spike
        self.spikecount = np.zeros(self.n)  # Spike counter

    def update(self, s, dt) -> None:
        """
        Update synapses given pre-synaptic spikes and a timestep
        :param s: presynaptic spikes (boolean)
        :param dt: timestep
        """

        # Spike vector to matrix considering synaptic strength

        for n in np.argwhere(s == 1):  # loop through spiking neurons
            if (
                self.spikecount[n] != 0
            ):  # Only update if neuron is not spiking for the first time

                decayR = np.exp(-self.deltaS[n] / self.tau_R[n])
                decayU = np.exp(-self.deltaS[n] / self.tau_U[n])

                self.R[n] = (
                    1 - (1 - self.R[n]) * decayR - self.R[n] * self.u[n] * decayR
                )
                self.u[n] = (
                    self.U[n]
                    + (self.u[n] + self.f[n] * (1 - self.u[n]) - self.U[n]) * decayU
                )

            self.deltaS[n] = 0  # set time since spike to zero
            self.spikecount[n] += 1  # add to spike counter

        # output
        self.out = self.u * self.R * s * self.w

        # increase time since spike counter
        self.deltaS += dt
        super().update(s, dt)

    def update_all(self, s, dt) -> None:
        """
        Update all synapses, irrespective of whether they spike or not.
        :param s: presynaptic spikes (boolean)
        :param dt: timestep
        """

        # pre-calculate decays
        decayR = np.exp(-self.deltaS / self.tau_R)
        decayU = np.exp(-self.deltaS / self.tau_U)

        # update synaptic mechanism of all neurons
        self.R_temp = 1 - (1 - self.R) * decayR - self.R * self.u * decayR
        self.u_temp = self.U + (self.u + self.f * (1 - self.u) - self.U) * decayU

        spiking = np.argwhere(s == 1).flatten()  # spiking neurons
        againspiking = spiking[
            np.argwhere(self.spikecount[spiking] > 0).flatten()
        ]  # neurons that don't spike for the first time
        neverspiking = np.argwhere(
            self.spikecount == 0
        )  # neurons that don't spike for the first time

        # accept update for neurons that spike for the second or more time
        self.R[againspiking] = self.R_temp[againspiking]
        self.u[againspiking] = self.u_temp[againspiking]

        # Reset update for neurons that have never spiked
        self.R_temp[neverspiking] = 1
        self.u_temp[neverspiking] = self.U[neverspiking]

        # record spike for all spiking neurons
        self.deltaS[spiking] = 0  # set time since spike to zero
        self.spikecount[spiking] += 1  # add to spike counter

        # output
        self.out = self.u * self.R * s

        # increase time since spike counter
        self.deltaS += dt
        super().update(s, dt)

    def run(self, spiketrain, dt, update_all=False) -> dict:
        """
        Computes synaptic output for a given input spike train.
        :param spiketrain: np.array of presynaptic spike trains at each time point
        :param dt: timestep
        :update_all: Boolean. If true, every synapse is updated at each time point and the mechanistic parameters
        (u, R, Z) are returned. If false, ODEs are integrated between spikes for efficient numerical implementation.
        """

        assert (
            spiketrain.shape[0] == self.n
        ), "Spike train does not correspond to number of synapses."

        # Time vector
        t = np.arange(0, spiketrain.shape[1] * dt, dt)
        assert len(t) == spiketrain.shape[1]
        steps = len(t)

        if update_all == True:

            self.u_temp = self.u.copy()
            self.R_temp = self.R.copy()

            # Initialize result matrices
            result = {
                "psr": np.zeros((self.n, steps)),
                "u": np.zeros((self.n, steps)),
                "R": np.zeros((self.n, steps)),
                "t": t,
            }

            # Loop over spike train
            for step in range(steps):
                self.update_all(s=spiketrain[..., step], dt=dt)
                result["psr"][..., step] = self.out
                result["u"][..., step] = self.u_temp
                result["R"][..., step] = self.R_temp

        else:
            # Initialize result matrices
            result = {"psr": np.zeros((self.n, steps)), "t": t}

            # Loop over spike train
            for step in range(steps):
                self.update(s=spiketrain[..., step], dt=dt)
                result["psr"][..., step] = self.out

        self.reset_()

        return result

    def reset_(self) -> None:
        """
        Resets synapses
        """
        self.u = self.U.copy()
        self.R = np.ones(self.n)
        self.deltaS = np.zeros(self.n)  # Time since previous spike
        self.spikecount = np.zeros(self.n)  # Spike counter

        super().reset_()


# Integrated between spikes for quick implementation. Input = ISI vector


def updateZ(Z, z, u, f, tau_Z, dt):

    return Z + (z + f * np.heaviside(u - z, 0) * (u - z) - Z) * np.exp(-dt / tau_Z)


def updateU(u, f, U, tau_U, dt):
    # increase of f * u * (1-u)
    return U + (u + f * u * (1 - u) - U) * np.exp(-dt / tau_U)


def updateR(r, u, z, tau_R, dt):
    return 1 - (1 - r) * np.exp(-dt / tau_R) - r * z * u * np.exp(-dt / tau_R)


def gTM_ISI(ISIvec, params):

    U, f, tau_U, Z, tau_Z, tau_R, A = params
    PSR_vec = []

    if np.isnan(Z):
        Z = U

    if np.isnan(A):
        A = 1 / (U * Z)

    for spike in range(len(ISIvec)):

        dt = ISIvec[spike]

        if spike == 0:
            u_new = U
            r_new = 1
            z_new = Z

        else:
            r_new = updateR(dt=dt, u=u_old, r=r_old, tau_R=tau_R, z=z_old)
            u_jump = u_old + f * u_old * (1 - u_old)
            u_new = updateU(dt=dt, u=u_old, U=U, f=f, tau_U=tau_U)
            z_new = updateZ(dt=dt, Z=Z, z=z_old, f=1, tau_Z=tau_Z, u=u_jump)

        PSR_vec.append(PSR(A, r_new, u_new, z_new))

        r_old = r_new
        u_old = u_new
        z_old = z_new

    return PSR_vec


def PSR(A, Rn, Un, Zn):
    """
    PostSynapticResponse function for the adapted TM network

    :param:     A (total efficacy in pA), U (utilized efficacy), R (available efficacy)

    :return:    estimated synaptic response (in pA)
    """

    return A * Rn * Un * Zn

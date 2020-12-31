from scipy.signal import *
import numpy as np
import time

# TESTING performance of four options of convolution

# General
dt = 0.1

# Kernel
tau = 50
amp = 1
T_kernel = 10 * tau
t = np.arange(0, T_kernel, dt)
kernel = amp * np.exp(-t / tau)

# Spiketrain
nspikes = 10000
st_length = 1000000
spiketrain = np.zeros(st_length)
spiketimes = np.random.randint(0, st_length, nspikes)
spiketrain[spiketimes] = 1

# OPTION 1: scipy.signal.lfilter
start = time.time()
res1 = lfilter(kernel, 1, spiketrain)
end = time.time()
dur1 = end - start


# OPTION 2: scipy.signal.sigtools._linear_filter
start = time.time()
res2 = sigtools._linear_filter(kernel, np.array([1]), spiketrain)
end = time.time()
dur2 = end - start


# OPTION 3: iterating over spiketrain


def convolve_add(x, a, axis=0):
    assert x.ndim == a.ndim, "x and a have to have equal number of dimensions!"
    assert axis <= x.ndim, "Arrays are not {} - dimensional !".format(axis)

    if x.shape[axis] < a.shape[axis]:
        return x + a[..., : x.shape[axis]]
    elif x.shape[axis] > a.shape[axis]:
        pad_dims = tuple(
            (0, x.shape[i] - a.shape[i]) if i == axis else (0, 0) for i in range(a.ndim)
        )
        return x + np.pad(a, pad_dims, "constant")
    else:
        return x + a


start = time.time()
length = np.min([len(spiketrain), len(kernel)])
res3 = np.zeros(spiketrain.shape)
for spiketime in np.where(spiketrain == 1)[0]:
    res3[spiketime + 1 :] += convolve_add(res3[spiketime + 1 :], kernel)
end = time.time()
dur3 = end - start

# OPTION 4: np.convolve

start = time.time()
res4 = np.convolve(kernel, spiketrain)[: len(spiketrain)]
end = time.time()
dur4 = end - start


print(dur1, dur2, dur3, dur4)

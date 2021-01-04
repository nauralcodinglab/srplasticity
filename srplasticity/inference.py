"""
inference.py

Everything related to parameter inference and fitting the model to data
"""

from srplasticity.srp import ExpSRP

ISIvec = [50] * 10

mu_taus = [10, 200, 650]
mu_amps = [9, -2, 11]
mu_baseline = -2.6
mu_scale = None

sigma_taus = mu_taus
sigma_amps = mu_amps
sigma_baseline = mu_baseline
sigma_scale = 1

model = ExpSRP(mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus, mu_scale, sigma_scale)

import time

start = time.time()
mean, sigma, _ = model.run_ISIvec(ISIvec, fast=True)
end = time.time()

print(end-start)

start = time.time()
mean2, sigma2, _, _ = model.run_ISIvec(ISIvec, fast=False)
end = time.time()

print(end-start)


scikits.pulsefit
================

A pulse-fitting library for python. 

## Overview

The `pulsefit` package provides functions allowing one to identify the
positions and amplitudes of a characteristic pulse shape within a
larger data array. It's main features are:

* It's robust when dealing with overlapping pulses.  
* It uses linear interpolation to provide non-integer pulse positions.
* It uses a fast moving-median filter to establish a baseline for
  determining the pulse amplitude.

The `pulsefit` package currently provides a single function,
`fit_mpoc_mle`. This function takes a fairly large number of
parameters that determine the fitting behavior. 

## Example usage

```Python
import numpy as np
#from scikits.pulsefit import fit_mpoc_mle
from scikits.pulsefit import fit_viewer


# Create a characteristic pulse shape. 
width = 10.0
tau = width / 2
x = np.arange(0, 10*width, dtype=np.float64)
p = (1*x/tau) * np.exp(-x/tau)
p /= p.max()


# Generate some noise.
sigma = 0.25
d = np.random.normal(2.0, sigma, 100000)


# Generate a uniform distribution of pulses. 
n_pulses = int(len(d) / 100)

inds0 = np.random.uniform(0, len(d) - 1, n_pulses)
inds0.sort()

amps0 = np.random.uniform(2.0, 5.0, n_pulses)

for (idx, amp) in zip(inds0, amps0):
    f_idx = np.floor(idx)
    xp = np.arange(len(p))
    x = xp - (idx % 1)
    di = min(len(p), len(d) - idx - 1)
    d[f_idx:f_idx + di] += amp * np.interp(x, xp, p)[:di]


# View fits. 
fit_viewer.view(
    d, p, 
    th=1.5,
    th_min=1.0, 
    filt_len=100, 
    pad_pre=20, 
    pad_post=50, 
    max_len=768, 
    min_denom=0.15, 
    exclude_pre=10, 
    exclude_post=50, 
    p_err_len=40, 
    sigma2=sigma*sigma, 
    chi2red_max=2.0, 
    abs_diff_max=1.25,
    correct=True, 
    pulse_add_len=70, 
    pulse_min_dist=2)
```

The `view` function has the same signature as the function `fit_mpoc_mle`. 

![Example](https://raw.githubusercontent.com/johnnylee/scikits.pulsefit/master/example/example.png)

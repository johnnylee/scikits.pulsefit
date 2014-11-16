scikits.pulsefit
================

A pulse-fitting library for python. 

## Overview

The `pulsefit` package provides functions allowing one to identify the positions and amplitudes of a characteristic pulse shape within a larger data array. It's main features are:

* It's robust when dealing with overlapping pulses.
* It uses linear interpolation to provide non-integer pulse positions.
* It uses a fast moving-median filter to establish a baseline for determining the pulse amplitude. 

The `pulsefit` package currently provides a single function, `fit_mpoc_mle`. This function takes a fairly large number of parameters that 

## Example usage


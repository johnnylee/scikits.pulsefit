from __future__ import print_function

import numpy as np
from scipy import optimize

import util


class OptFitLeastSq(object):
    """OptFitLeastSq optimizes the positions and amplitudes of the
    identified pulses.
    """
    def __init__(self, ampfit, debug=False):
        """Arguments: 
        ampfit   -- An object witha fit function for fitting pulse
                    amplitudes. 
        """
        self.ampfit = ampfit
        self.debug = debug

        
    def optimize(self, block):
        if self.debug:
            print("\nOptimizing fit...")
        
        # What we're trying to fit. 
        data = block.r[block.i0:block.i1]

        # Various parameters that we'll need.
        N = len(block.inds)
        exclude_pre = block.exclude_pre
        idx_max = block.i1 - block.i0 - block.exclude_pre - block.exclude_post
        model = block.model
        b = block.b
        p = block.p

        def model_fn(args):
            inds = (args[:N] % idx_max) + exclude_pre
            amps = args[N:]
            util.model(b, inds, amps, p, model)
            return model
            
        def err_fn(args):
            return data - model_fn(args)
            
        def scalar_err_fn(args):
            return np.sum(err_fn(args)**2)

        # Get initial guess for best-fit parameters. 
        x0 = np.concatenate((block.inds - exclude_pre, block.amps))

        xopt, cov_x, infodict, mesg, ier = optimize.leastsq(
            err_fn, x0, full_output=True)
        
        if self.debug and ier not in (1, 2, 3, 4):
            print("    Optimization failed: " + mesg)

        block.inds = (xopt[:N] % idx_max) + exclude_pre
        block.amps = xopt[N:]

        # Store the best fit model. 
        util.model(b, block.inds, block.amps, p, model)
        util.residual(block)

        if self.debug:
            if ier not in (1, 2, 3, 4):
                print("    Optimization failed: " + mesg)
            print("    Inds:", block.inds)
            print("    Amps:", block.amps)

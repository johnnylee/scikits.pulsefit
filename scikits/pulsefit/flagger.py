from __future__ import print_function

import numpy as np


class Flagger(object):
    """Flagger flags pulses based on the value of reduced chi^2."""
    def __init__(self, p_err_len, sigma2, chi2red_max, debug=False):
        """Arguments:
        p_err_len    -- The length along each pulse used to compute the
                        reduced chi^2 and residual.
        sigma2       -- The variance in the data.
        chi2red_max  -- Maximum acceptable reduced chi^2.
        """
        self.p_err_len = p_err_len
        self.sigma2 = sigma2
        self.chi2red_max = chi2red_max
        self.debug = debug

    def flag(self, block):
        if self.debug:
            print("\nFlagging...")

        # The number of degrees of freedom. This is equal to
        # (# observations) - (# fitted parameters) - 1
        # For each pulse, there are two fitted parameters: the
        # offset and the amplitude.
        nu = self.p_err_len - 3

        block.flags = np.empty(block.inds.size, dtype=np.uint8)
        block.chi2red = np.empty_like(block.amps)

        for i, idx1 in enumerate(block.inds):
            res = block.res[idx1:idx1 + self.p_err_len]
            block.chi2red[i] = np.sum(res**2) / (nu * self.sigma2)
            if block.chi2red[i] > self.chi2red_max:
                block.flags[i] = 1
            elif block.amps[i] <= 0:
                block.flags[i] = 2
            else:
                block.flags[i] = 0

        if self.debug:
            for flag in block.flags:
                if flag == 0:
                    print("    OK")
                elif flag == 1:
                    print("    Reduced chi^2 too large:", max(block.chi2red))
                elif flag == 2:
                    print("    Negative amplitude:", max(block.amps))
                else:
                    print("    Unknown error.")

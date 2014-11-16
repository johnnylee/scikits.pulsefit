
import numpy as np


class Flagger(object):
    """Flagger flags pulses based on two metrics: 
    1) The maximum absolute value of the residual.
    2) The value of reduced chi^2.
    """
    def __init__(self, p_err_len, sigma2, chi2red_max, abs_diff_max,
                 debug=False):
        """Arguments:
        p_err_len    -- The length along each pulse used to compute the
                        reduced chi^2 and residual.
        sigma2       -- The variance in the data.
        chi2red_max  -- Maximum acceptable reduced chi^2.
        abs_diff_max -- The maximum acceptable absolute difference
                        in the residual over the pulse.
        """
        self.p_err_len = p_err_len
        self.sigma2 = sigma2
        self.chi2red_max = chi2red_max
        self.abs_diff_max = abs_diff_max

    
    def flag(self, block):
        # The number of degrees of freedom. This is equal to
        # (# observations) - (# fitted parameters) - 1
        # For each pulse, there are two fitted parameters: the 
        # offset and the amplitude. 
        nu = self.p_err_len - 3
        
        block.flags = np.empty(block.inds.size, dtype=np.uint8)
        block.chi2red = np.empty_like(block.amps)
        block.res_max = np.empty_like(block.amps)
        
        for i, idx1 in enumerate(block.inds):
            res = block.res[idx1:idx1 + self.p_err_len]
            block.chi2red[i] = np.sum(res**2) / (nu * self.sigma2)
            block.res_max[i] = max(res.max(), -res.min())
            if block.res_max[i] > self.abs_diff_max:
                block.flags[i] = 1
            elif block.chi2red[i] > self.chi2red_max:
                block.flags[i] = 2
            else:
                block.flags[i] = 0

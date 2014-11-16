from __future__ import print_function

import numpy as np
import ampfit_mle_c

class AmpFitMLE(object):
    """AmpFitMLE fits pulse amplitudes using a maximum-likelihood
    estimation.
    """
    def __init__(self, debug=False):
        self.debug = debug
        
    
    def _compute_lambda_matrix(self, n_r, p, inds):
        """Compute the lambda matrix given the block size, n."""
        lam = np.empty((inds.size, inds.size), dtype=np.float64)
        ampfit_mle_c.compute_lambda_matrix(n_r, p, inds, lam.reshape(-1))
        return lam


    def _compute_phi_array(self, r, p, inds, b):
        phi = np.empty(inds.size, dtype=np.float64)
        ampfit_mle_c.compute_phi_array(r - b, p, inds, phi)
        return phi

    
    def fit(self, block):
        """Find the best-fit amplitude for pulses whose positions have been
        identified. 
        """
        if self.debug:
            print("\nFinding pulse amplitudes...")

        # No pulses - nothing to do.
        if len(block.inds) == 0:
            block.amps = np.empty(0, dtype=np.float64)
            return

        r = block.r[block.i0:block.i1]
        p = block.p
        
        # Compute the lambda matrix and phi array.
        lam = self._compute_lambda_matrix(r.size, p, block.inds)
        phi = self._compute_phi_array(r, p, block.inds, block.b)
        
        # Create a separate fast-path for single pulses.
        if block.inds.size == 1:
            if lam[0][0] == 0:
                block.amps = np.zeros_like(block.inds)
            else:
                block.amps = np.array(
                    (phi[0] / lam[0][0],), dtype=np.float64)
        # Otherwise we use linalg.solve for multiple pulses. 
        else:
            try:
                block.amps = np.linalg.solve(lam, phi)
            except Exception, ex:
                if self.debug:
                    print("    Error:", ex)
                # This occurs when we have a singular matrix. 
                block.amps = np.zeros_like(block.inds)
            
        if self.debug:
            print("    Amps:", block.amps)

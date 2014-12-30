from __future__ import print_function

import numpy as np


class PulseIdentMPOC(object):
    """PulseIdentMPOC identifies the indices of pulses within blocks."""
    def __init__(self, th, min_denom=0.01, debug=False):
        """Arguments:
        th          -- Threshold value for the smallest pulse.
        min_denom   -- The minimum allowed fractional FFT denominator
                       value.
        debug       -- If True, save `z` as "mpoc_z" in block.kwinfo.
        """
        self.th = float(th)
        self.min_denom = float(min_denom)
        self.debug = debug

    def find_pulses(self, block, set_z=False):
        """Find the indices of pulses in the given block."""
        if self.debug:
            print("\nFinding pulse indices...")

        r = block.r[block.i0:block.i1] - block.b
        p = block.p
        n = max(r.size, p.size)

        R = np.fft.rfft(r, n)
        P = np.fft.rfft(p, n)

        denom = np.abs(P)

        min_denom = self.min_denom * denom.max()
        denom[denom < min_denom] = min_denom

        Z = R * P.conjugate() / denom

        # Get correlation by inverse fft.
        z = np.fft.irfft(Z)[:r.size]

        # Scale correlation to measured pulse height.
        z /= z.max()
        z *= r.max()

        # Find correlation peaks - local maxima above the threshold.
        inds = np.where((z[1:-1] >= z[:-2]) &
                        (z[1:-1] > z[2:]) &
                        (z[1:-1] > self.th))[0] + 1

        # Ignore peaks in the excluded regions.
        inds = inds[((inds > block.exclude_pre) &
                     (inds < r.size - block.exclude_post))]

        # Return at least one pulse.
        if inds.size != 0:
            block.inds = inds
        else:
            block.inds = np.array((z.argmax(),), dtype=np.int64)

        if self.debug:
            print("    Inds:", block.inds)
            block.kwinfo["z"] = z

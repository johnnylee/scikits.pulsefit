from __future__ import print_function

import numpy as np


class CorrectAddPulses(object):
    def __init__(self, ampfit, optfit, flagger, pulse_add_len, th_min,
                 min_dist=0, debug=False):
        """Arguments:
        ampfit   -- Ampitude fitter.
        optfit   -- Fit optimizer.
        flagger  -- Block flagger.
        pulse_add_len -- The max number of pulses to add is the block
                         length divided by pulse_add_len.
        th_min   -- Minimum allowed pulse height. 
        min_dist -- If pulses are closer than min_dist, they are merged
                    into a single pulse.
        """
        self.ampfit        = ampfit
        self.optfit        = optfit
        self.flagger       = flagger
        self.pulse_add_len = pulse_add_len
        self.th_min        = th_min
        self.min_dist      = min_dist
        self.debug         = debug


    def sanitize(self, block):
        refit = False
        
        # Remove low-amplitude pulses. 
        mask = block.amps > self.th_min
        if mask.sum() != block.inds.size:
            refit = True
            block.inds = block.inds[mask]
            block.amps = block.amps[mask]
            if self.debug:
                print("Correct: Removed low-amplitude pulses.")

        # Merge pulses that are too close together.
        if self.min_dist != 0 and len(block.inds) > 1:
            dinds = block.inds[1:] - block.inds[:-1]
            mask = dinds > self.min_dist
            if mask.sum() != block.inds.size - 1:
                refit = True
                new_inds = np.empty(mask.sum() + 1, dtype=np.float64)
                new_inds[0] = block.inds[0]
                if mask.sum() != 0:
                    new_inds[1:] = block.inds[1:][mask]

                if self.debug:
                    print("Correct: Merged pulses.")
            
        if refit:
            self.refit(block)
            

    def refit(self, block):
        self.ampfit.fit(block)
        self.optfit.optimize(block)
        self.flagger.flag(block)
        

    def correct(self, block):
        if self.debug:
            print("\nCorrecting...")

        add_max = int((block.i1 - block.i0) / self.pulse_add_len)

        for i in xrange(add_max):
            if np.all(block.flags == 0):
                return
                
            # Add a new pulse. 
            idx_new = max(block.res.argmax() - block.p.argmax(), 0)
            inds = np.concatenate((block.inds, (idx_new,)))
            inds.sort()
            block.inds = inds

            if self.debug:
                print("    Adding pulse at:", idx_new)
                print("    Inds:", block.inds)

            self.refit(block)
            self.sanitize(block)

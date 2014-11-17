from __future__ import print_function 

import numpy as np

from blockident_median import BlockIdentMedian
from pulseident_mpoc import PulseIdentMPOC
from ampfit_mle import AmpFitMLE
from optfit_leastsq import OptFitLeastSq
from flagger import Flagger
from correct_addpulses import CorrectAddPulses
import util

def fit_mpoc_mle(
        r, p, 
        th, th_min, filt_len, pad_pre, pad_post, max_len, 
        min_denom, exclude_pre, exclude_post, 
        p_err_len, sigma2, chi2red_max, 
        correct=True, pulse_add_len=None, pulse_min_dist=0,
        return_blocks=False, debug=False, cb=None):
    """fit_mpoc_mle
    
    Fit pulses using a modified phase-only correlation to identify pulse
    positions and a maximum-likelihood estimate for amplitudes. 

    Description of operation:
    
    * Blocks containing pulses are identified using a fast moving
      median filter in combination with a threshold value.

    * The position of pulses within a block are determined using 
      the MPOC algorithm. 

    * The amplitudes of pulses within a block are determined using 
      a maximum likelihood estimation. 

    * The positions and amplitudes of pulses are refined using the
      Levenberg-Marquardt algorithm scipy.optimize.leastsq. Linear
      interpolation is used to allow non-integer pulse positions. 

    * Correctly fit pulses are subtracted from the residual, `r`. If
      there are additional pulses in the block, an additional fit is
      attempted.
    
    * Additional pulses are optionally added to each block in an
      attempt to correct errors. 

    Parameters 
    ---------- 
    r : float64 array 
        The raw data. This will be modified in place.  
    p : float64 array 
        The pulse shape.

    th : float
        The lowest pulse amplitude of interest.
    th_min : float
        The lowest pulse amplitude that will be kept or fit.
    filt_len : int
        The length of the moving-median filter used to establish
        the zero-level for block-identification.
    pad_pre : int
        Padding below `th` added to the start of each block. 
    pad_post : int
        Like `pad_pre`, but added at the end of each block.
    max_len : int
        The maximum block length produced. 

    min_denom : float
        The minimum denominator used in the MPOC routine 
        identifying pulse positions. 
    exclude_pre : int
        Pulses within this number of sample from the beginning
        of each block will be ignored. 
    exclude_post : int
        Like `exclude_pre`, but at the end of each block. 
    
    p_err_len : int
        The number of samples to check for each pulse to determine
        the reduced chi^2.
    sigma2 : float 
        The variance in the noise. This is used to compute the 
        reduced chi^2. 
    chi2red_max : float
        The maximum acceptable reduced chi^2 value for a pulse fit. 
    
    correct : bool
        If True, attempt to correct errors. 
    pulse_add_len : int
        The number of pulses the correction code will attempt to 
        add in order to correct a failed fit is 
        (block length) / `pulse_add_len`. 
    pulse_min_dist: int
        Pulses closer than this many samples apart will be merged 
        when attempting correction. 
    
    return_blocks : bool
        If True, return a list of Block objects instead of 
        concatenating and sorting the identified pulses.
    
    cb : function(block) 
        A function taking a block called after each block is
        identified.

    Returns
    -------
    If return_blocks is False (default), return 
        (inds, amps, flags, chi2red) 
        inds    : Indices of located pulses. 
        amps    : Amplitudes of located pulses. 
        flags   : A flag for each pulse. 0 means successfully fit. 
        chi2red : The reduced chi^2 value for each pulse. 

    If return_blocks is True, return a list of fit blocks.

    """
    if debug:
        np.set_printoptions(precision=2)

    # The block-identification routine. 
    block_ident = BlockIdentMedian(
        r, p, th, filt_len, pad_pre, pad_post, max_len, 
        exclude_pre, exclude_post, debug=debug)
    
    # The pulse-identification routine. 
    pulse_ident = PulseIdentMPOC(th, min_denom, debug=debug)

    # The amplification-fitting routine. 
    amp_fit = AmpFitMLE(debug=debug)
    
    # The fit-optimization routine. 
    opt_fit = OptFitLeastSq(amp_fit, debug=debug)
        
    # The flagger. 
    flagger = Flagger(p_err_len, sigma2, chi2red_max, debug=debug)
        
    # The error-correction routine. 
    corrector = None
    if correct:
        corrector = CorrectAddPulses(
            amp_fit, opt_fit, flagger, pulse_add_len, pulse_min_dist, 
            debug=debug)

    # Find and fit blocks. 
    blocks = []
    while 1:
        
        # Get the next block. 
        block = block_ident.next_block()
        if block is None:
            break
            
        # Find and fit pulses. 
        pulse_ident.find_pulses(block)
        amp_fit.fit(block)
        
        # Optimize the fit. 
        opt_fit.optimize(block)
            
        # Flag. 
        flagger.flag(block)
        
        # All pulses found were flagged bad, attempt correction. 
        if corrector is not None and np.all(block.flags != 0):
            corrector.correct(block)

        # If some but not all pulses were fit correctly, we
        # keep the good fits in the block and rewind the block-
        # identification algorithm. 
        if not (np.all(block.flags == 0) or np.all(block.flags != 0)):
            if debug:
                print("Rewinding to:", block.i0)
            util.remove_bad_pulses(block)
            block_ident.set_position(block.i0)
            
        # When debugging, save the residual for review. 
        if debug:
            block.kwinfo['r'] = r[block.i0:block.i1].copy()

        # Subtract pulses that fit properly. 
        util.subtract_pulses(block)
        
        # Save the block. 
        blocks.append(block)
        
        if cb is not None:
            cb(block)
            
    if return_blocks:
        return blocks

    # Create and sort output arrays. 
    inds = np.concatenate([b.inds + b.i0 for b in blocks])
    amps = np.concatenate([b.amps for b in blocks])
    flags = np.concatenate([b.flags for b in blocks])
    chi2red = np.concatenate([b.chi2red for b in blocks])

    mask = inds.argsort()
    
    return inds[mask], amps[mask], flags[mask], chi2red[mask]

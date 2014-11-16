
import numpy as np

class Block(object):
    """A Block is a segment of data that may contain pulses. Blocks
    are identified and then passed to an algorithm that attempts to
    identify pulses in the block.
    
    A Block contains the following data members:
    r   - Raw data (full).
    p   - Pulse shape. 
    
    i0 - The starting index of the block in the data stream.
    i1 - The final index of the block in the data stream.
    b  - The best fit offset for the block.
    
    exclude_pre  - The number of samples at the beginning and end
    exclude_post - of the block that should be free of pulses. 
    
    inds    - The indices of identified pulses.
    amps    - The amplitudes of identified pulses.
    flags   - Numeric flags set by the fitting algorithm, per pulse.
              0 means no error.
    chi2red - The reduced chi^2 statistic for the fit.
    res_max - The maximum absolute value in the residual.

    model   - An array containing the model data. 
    res     - An array containing the residual data. 
    
    kwinfo  - A dictionary that can be used to attach 
    """
    def __init__(self, r, p, i0, i1, b):
        self.r  = r
        self.p  = p
        self.i0 = i0
        self.i1 = i1
        self.b  = b
        
        self.exclude_pre  = 0
        self.exclude_post = 0

        self.inds    = None
        self.amps    = None
        self.flags   = None
        self.chi2red = None
        self.res_max = None

        self.model   = np.empty(i1 - i0, dtype=np.float64)
        self.res     = np.empty(i1 - i0, dtype=np.float64)

        self.kwinfo = {}

        

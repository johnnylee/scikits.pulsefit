

from util_c import add_pulses


def model(b, inds, amps, p, out):
    out[:] = b
    add_pulses(inds, amps, p, out)


def residual(block):
    block.res[:] = block.r[block.i0:block.i1] - block.model


def subtract_pulses(block):
    """Subtract pulses in block from r."""
    mask = block.flags == 0
    inds = block.inds[mask] + block.i0
    amps = block.amps[mask] * -1
    add_pulses(inds, amps, block.p, block.r)


def remove_bad_pulses(block):
    """Modifies the block, keeping only good pulses."""
    mask = block.flags == 0
    block.inds = block.inds[mask]
    block.amps = block.amps[mask]
    block.flags = block.flags[mask]
    block.chi2red = block.chi2red[mask]
    
    # Recompute the model and residual. 
    model(block.b, block.inds, block.amps, block.p, block.model)
    residual(block)

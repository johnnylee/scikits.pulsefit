from __future__ import print_function

import matplotlib.pyplot as plt
from fit_mpocmle import fit_mpoc_mle


def view(
        r, p, 
        th, th_min, filt_len, pad_pre, pad_post, max_len, 
        min_denom, exclude_pre, exclude_post, 
        p_err_len, sigma2, chi2red_max, 
        correct=True, pulse_add_len=None, pulse_min_dist=0,
        return_blocks=False, cb=None):

    plt.interactive(True)

    def _view_block(b):
        fig = plt.gcf()
        fig.clf()
        ax = fig.gca()
        
        ax.plot(b.kwinfo['r'] - b.b, 'b.-', label='Raw Data')
        ax.plot(b.res, 'k.-', label='Residual')
        ax.plot(b.model - b.b, 'r.-', label='Model')
        ax.plot(b.inds + p.argmax(), b.amps, 'rD', label="Peaks")
        ax.plot(b.kwinfo['z'], 'y-', label='z')
        ax.axhline(th, color='m', label='Threshold')
        ax.axhline(-th, color='m')
        
        kwargs = { 'facecolor': '0.75', 'alpha':0.30 }
        ax.axvspan(0, exclude_pre, **kwargs)
        ax.axvspan(len(b.res) - exclude_post, len(b.res), **kwargs)
        
        ax.axhspan(-th_min, th_min, facecolor='g', alpha=0.15)
        
        ax.set_title("Index: {0}".format(b.i0))
        
        ax.grid(True)
        ax.legend()
        plt.draw()
        
        if cb is not None:
            cb(block)
            
        print("\nBlock:")
        print("    Inds          :", b.inds)
        print("    Amps          :", b.amps)
        print("    Flags         :", b.flags)
        print("    Reduced chi^2 :", b.chi2red)
        
        print("\nPress enter to continue...")
        raw_input()
        print("-" * 78)
    
    return fit_mpoc_mle(
        r, p, 
        th, th_min, filt_len, pad_pre, pad_post, max_len, 
        min_denom, exclude_pre, exclude_post, 
        p_err_len, sigma2, chi2red_max, 
        correct, pulse_add_len, pulse_min_dist,
        return_blocks=return_blocks, debug=True, cb=_view_block)



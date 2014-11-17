#! /usr/bin/env python

import os
import sys

DISTNAME            = 'scikits.pulsefit'
DESCRIPTION         = 'Scikits pulse-fitting package.'
LONG_DESCRIPTION    = """
Pulse-fitting library for identifying the positions and amplitudes of
a characteristic pulse shape.
"""
MAINTAINER          = 'J. David Lee',
MAINTAINER_EMAIL    = 'johnl@crumpington.com',
URL                 = 'https://github.com/johnnylee/scikits.pulsefit'
LICENSE             = 'MIT'
DOWNLOAD_URL        = URL
VERSION             = '0.1.1'

import setuptools
from numpy.distutils.core import setup, Extension

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version = VERSION,
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
          install_requires = 'numpy',
          namespace_packages = ['scikits'],
          packages = setuptools.find_packages(),
          ext_modules = [
              Extension(
                  'scikits.pulsefit.blockident_median_c', 
                  ['src/blockident_median.c']),
              Extension(
                  'scikits.pulsefit.ampfit_mle_c', 
                  ['src/ampfit_mle.c']),
              Extension(
                  'scikits.pulsefit.util_c', 
                  ['src/util.c']),
          ],
          include_package_data = True,
          zip_safe = True, # the package can run out of an .egg file?
          classifiers =
          [ 'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering'])

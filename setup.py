#! /usr/bin/env python

from distutils.core import setup, Extension
import numpy

setup(
    name='scikits.pulsefit',
    version='0.1.3',
    description='Characteristic pulse fitting.',

    author='J. David Lee',
    author_email='johnl@crumpington.com',

    maintainer='Crumpington Consulting LLC',
    maintainer_email='johnl@crumpington.com',
    
    url='https://github.com/johnnylee/scikits.pulsefit',
    
    packages=['scikits/pulsefit'],
    
    ext_modules=[
        Extension(
            'scikits.pulsefit.blockident_median_c', 
            ['src/blockident_median.c'],
            include_dirs=[numpy.get_include()]),
        Extension(
            'scikits.pulsefit.ampfit_mle_c', 
            ['src/ampfit_mle.c'],
            include_dirs=[numpy.get_include()]),
        Extension(
            'scikits.pulsefit.util_c', 
            ['src/util.c'],
            include_dirs=[numpy.get_include()]),
    ],
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    
    license='MIT'
)


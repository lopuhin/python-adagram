#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Extension

# FIXME - ehhh how do we build it?
from Cython.Build import cythonize
import numpy as np


setup(name='adagram',
      version='0.0.1',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'cython',
          'joblib',
          'numpy>=1.9',
          'six',
      ],
      ext_modules=cythonize([
          Extension(
              'adagram/*', ['adagram/*.pyx'],
              include_dirs=[np.get_include()],
              extra_compile_args=['-march=native', '-O3', '-ffast-math']
          ),
          ]),
      entry_points={
          'console_scripts': [
              'adagram-train = adagram.train:main',
          ]
      },
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(name='adagram',
      version='0.0.1',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'statprof==0.1.2',
          'cffi>=1.0',
          'numpy==1.9', # PyPy: git+https://bitbucket.org/pypy/numpy.git
      ],
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(name='adagram',
      version='0.0.1',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'cffi>=1.0',
          'joblib',
          'numpy>=1.9',
          'six',
      ],
      entry_points={
          'console_scripts': [
              'adagram-train = adagram.train:main',
          ]
      },
)

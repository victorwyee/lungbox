#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: setup.py
"""
To build:
>> pip install -r requirements.txt
>> python setup.py install
"""

from os import path
import setuptools

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setuptools_version = int(setuptools.__version__.split('.')[0])
assert setuptools_version > 30, "Installation requires setuptools > 30"

this_dirpath = path.abspath(path.dirname(__file__))

with open(path.join(this_dirpath, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')


REQUIRED_PACKAGES = [
    'atlas==0.27.0',
    'boto3==1.7.82',
    'botocore==1.10.84',
    'configs==3.0.3',
    'cython==0.28.4',
    'h5py==2.8.0',
    'imgaug==0.2.6',
    'IPython[all]',
    'Keras==2.1.3',            # working multi-GPU training
    'Keras-Applications==1.0.4',
    'Keras-Preprocessing==1.0.2',
    'matplotlib==3.0.0',
    'numpy==1.14.5',
    'opencv-python==3.4.3.18',
    'pandas==0.23.4',
    'Pillow==5.2.0',
    'pydicom==1.1.0',
    'scikit-image==0.13',      # avoid antialiasing warnings in 0.14, enough functionality
    'scipy==1.1.0',
    'tensorflow==1.10.0',
    'tensorflow-gpu==1.10.0',
    'tensorboard==1.10.0',
    'tornado==4.5.3'           # allows async within async
]

setup(
    name='lungbox',
    version='0.0.1',
    url='https://github.com/victorwyee/lungbox',
    author='Victor W. Yee',
    author_email='w.victoryee@gmail.com',
    license='MIT',
    description='A bounding box predictor for pneumonia from chest x-rays',
    long_description=long_description,
    python_requires='>=3.6',
    install_requires=REQUIRED_PACKAGES,
    tests_require=['flake8', 'scikit-image'],
)

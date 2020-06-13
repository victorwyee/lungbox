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

def _parse_requirements(fpath):
    with open(fpath) as f:
        return f.read().splitlines()

EXTRA_PACKAGES = {
    "tf": ["tensorflow==1.15.2"],
    "tf_gpu": ["tensorflow-gpu==1.15.2"],
}

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
    install_requires=_parse_requirements('requirements.txt'),
    extras_require=EXTRA_PACKAGES,
    tests_require=['flake8', 'scikit-image'],
)

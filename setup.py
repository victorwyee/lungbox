#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: setup.py
"""
To build:
>> pip install -r requirements.txt
>> python setup.py install
"""

from os import path
import pip
import logging
import pkg_resources
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


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name='lungbox',
    version='0.0.1',
    url='https://github.com/victorwyee/lungbox',
    author='Victor W. Yee',
    author_email='w.victoryee@gmail.com',
    license='MIT',
    description='A bounding box predictor for pneumonia from chest x-rays',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    install_requires=install_reqs,
    tests_require=['flake8', 'scikit-image'],
)

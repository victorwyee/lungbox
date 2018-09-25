import setuptools
from setuptools import setup
from os import path
import platform

version = int(setuptools.__version__.split('.')[0])
assert version > 30, "Installation requires setuptools > 30"

this_dirpath = path.abspath(path.dirname(__file__))

with open(path.join(this_dirpath, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')

setup(
    name='lungbox',
    version=__version__,   # noqa
    description='A bounding box predictor for pneumonia from chest x-rays',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "atlas==0.27.0",
        "boto3==1.7.82",
        "botocore==1.10.84",
        "configs==3.0.3",
        "imgaug==0.2.6",
        "matplotlib==3.0.0",
        "numpy==1.14.5",
        "pandas==0.23.4",
        "pydicom==1.1.0",
        "scikit-image==0.13",  # older version to avoid massive antialiasing warnings
        "streamlit==0.15.5",
        "tornado==4.5.3"       # allows async within async
    ],
    tests_require=['flake8', 'scikit-image'],
)

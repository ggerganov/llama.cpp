
from setuptools import setup, find_packages
import glob, os

setup(
    name='llama_cpp',
    version='0.0.1',
    author='Anonymous',
    author_email='',
    license='All rights reserved',
    packages=find_packages(where='py'),
    package_dir={'': 'py'},
    install_requires=[],
    entry_points={'console_scripts': []},
)

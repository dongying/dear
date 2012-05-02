#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name='dear',
    version='0.1.1',
    description='Dear EARs Audio Analysis Framework',
    author='Dongying Zhang',
    author_email='zhdongying@gmail.com',
    url='http://dongying.github.com/dear',
    packages=['dear','dear.spectrum','dear.analysis', 'dear.io'],
    license='MIT LICENSE',
)


#-*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("cqt", ["dear/spectrum/cqt.pyx"], ['.',numpy.get_include()])
]

setup(
    name = 'cydear',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)


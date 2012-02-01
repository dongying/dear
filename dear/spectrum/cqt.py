#-*- coding: utf-8 -*-

from _base import SpectrumBase
import numpy


class Spectrum(SpectrumBase):
    '''Spectrum of Constant-Q Transform'''

    @staticmethod
    def pre_calculate(win, win_shape):
        pass

    @staticmethod
    def transform(samples, win_shape=numpy.hamming, pre_var=None):
        pass

    def walk(self, Q, win, step, start=0, end=None, join_channels=True,
            win_shape=numpy.hamming):
        var = self.pre_caculate(win, win_shape)

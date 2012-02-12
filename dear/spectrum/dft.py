#-*- coding: utf-8 -*-

from _base import *
import numpy


class Spectrum(SpectrumBase):
    '''Spectrum of Discrete Fourier Transform'''
    
    @staticmethod
    def pre_calculate(win, win_shape, pre=True):
        var = {
            'WL': win,
            'W': win_shape(win)}
        #
        if pre:
            arr = 2. * numpy.pi * numpy.arange(win) / win
            PRE = numpy.array([var['W'] * (numpy.cos(arr*k) - numpy.sin(arr*k)*1j) \
                    for k in xrange(win/2+1)])
            var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform_pre(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape, True)
        frame = samples * pre_var.PRE
        frame = frame.sum(1) / pre_var.WL
        return frame

    @staticmethod
    def transform(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape, False)
        return numpy.fft.rfft(pre_var.W * samples) / pre_var.WL

    def walk(self, win=1024, step=512, start=0, end=None, join_channels=True,
            win_shape=numpy.hamming, mpre=False):
        var = self.pre_calculate(win, win_shape, mpre)
        self._var = var
        transform = mpre and self.transform_pre or self.transform
        #
        for samples in self.audio.walk(win, step, start, end, join_channels):
            if join_channels:
                yield transform(samples, pre_var=var)
            else: yield [transform(ch, pre_var=var) for ch in samples]


class PowerSpectrum(Spectrum):

    @staticmethod
    def transform_pre(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape, True)
        frame = samples * pre_var.PRE
        frame = frame.sum(1)
        return (frame.real**2 + frame.imag**2) / pre_var.WL

    @staticmethod
    def transform(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape, False)
        frame = numpy.fft.rfft(pre_var.W * samples)
        return (frame.real**2 + frame.imag**2) / pre_var.WL


#-*- coding: utf-8 -*-

from _base import SpectrumBase
import numpy


class Spectrum(SpectrumBase):
    '''Spectrum of Discrete Fourier Transform'''
    
    @staticmethod
    def pre_calculate(win, win_shape):
        var = {
            'WL': win,
            'W': win_shape(win)}
        #
        PRE = []
        arr = 2. * numpy.pi * numpy.arange(win) / win
        for k in xrange(win/2+1):
            PRE.append(
                var['W'] * (numpy.cos(arr*k) - numpy.sin(arr*k)*1j) 
            )
        var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape)
        frame = numpy.array(
                [numpy.sum(samples * pre) for pre in pre_var.PRE])
        return frame / pre_var.WL

    def walk(self, win=1024, step=512, start=0, end=None, join_channels=True,
            win_shape=numpy.hamming):
        var = self.pre_calculate(win, win_shape)
        #
        if join_channels:
            for samples in self.audio.walk(win, step, start, end, 
                    join_channels):
                yield self.transform(samples, pre_var=var)
        else:
            for samples in self.audio.walk(win, step, start, end, 
                    join_channels):
                frame = []
                for ch in samples:
                    frame.append(self.transform(ch, pre_var=var))
                yield frame


#-*- coding: utf-8 -*-

from _base import SpectrumBase
import numpy


class Spectrum(SpectrumBase):
    '''Spectrum of Discrete Fourier Transform'''
    
    def _pre_calculate(self, win, step, win_shape):
        var = {
            'WL': win,
            'SL': step,
            'SR': self.audio.samplerate,
            'D': self.audio.duration,
            'CHN': self.audio.channels,
            'W': win_shape(win)}
        #
        PRE = []
        arr = 2. * numpy.pi * numpy.arange(win) / win
        for k in xrange(win/2):
            PRE.append(
                var['W'] * (numpy.cos(arr*k) - numpy.sin(arr*k)*1j) 
            )
        var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    def walk(self, win=1024, step=512, start=0, end=None, join_channels=True, 
            win_shape=numpy.hamming):
        ''''''
        var = self._pre_calculate(win, step, win_shape)
        #
        if join_channels:
            for samples in self.audio.walk(win, step, start, end, 
                    join_channels):
                frame = numpy.array(
                        [numpy.sum(samples * pre) for pre in var.PRE])
                yield frame / var.WL
        else:
            for samples in self.audio.walk(win, step, start, end, 
                    join_channels):
                frame = []
                for ch in samples:
                    data = numpy.array(
                            [numpy.sum(ch * pre) for pre in var.PRE])
                    frame.append(data / var.WL)
                yield frame


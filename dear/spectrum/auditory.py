#-*- coding: utf-8 -*-

from _base import *
from cqt import CNTPowerSpectrum, A0, C8
import numpy
import scipy.signal as sig

inf = float('inf')


class Y1(CNTPowerSpectrum):
    
    def walk(self, N=24, freq_base=A0, freq_max=C8, hop=0.01, start=0, 
            end=None, win_shape=numpy.hamming, resize_win=True):
        parent = super(Y1, self)
        for vector in parent.walk(N,freq_base,freq_max,hop,start,end,
                join_channels=True,win_shape=win_shape,resize_win=resize_win):
            v = numpy.maximum(vector, PW_MIN)
            yield 20 * numpy.log10(v / PWI_THS)


class Y2(Y1):
    
    @staticmethod
    def g(vector, gamma=10):
        if not gamma or gamma >= inf:
            return 0.5
        return 1./(1+numpy.exp(-gamma*vector)) - 0.5

    def walk(self, gamma=10, *args, **kw):
        def gnt():
            vpre = 0
            for vector in super(Y2, self).walk(*args, **kw):
                v = self.g(vector - vpre)
                yield v
                vpre = vector
        wb,wa = sig.butter(6, 9000./self.audio.samplerate)
        spectro = [v for v in gnt()]
        for v in sig.lfilter(wb,wa,spectro,0):
            yield v
        

class Y3(Y2):
    def walk(self, *args, **kw):
        for vector in super(Y3, self).walk(*args, **kw):
            yield numpy.append(vector[0], numpy.diff(vector,1))


class Y4(Y3):
    def walk(self, *args, **kw):
        for vector in super(Y4, self).walk(*args, **kw):
            yield numpy.maximum(vector, 0)


class Y5(Y4):
    def walk(self, *args, **kw):
        pass


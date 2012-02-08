#-*- coding: utf-8 -*-

from _base import SpectrumBase
from cqt import CNTSpectrum, A0, A8
import numpy
import scipy.signal as sig

inf = float('inf')


class Y1(CNTSpectrum):
    
    def walk(self, N=12, freq_base=A0, freq_max=A8, hop=0.02, start=0, 
            end=None, win_shape=numpy.hamming):
        parent = super(Y1, self)
        n, k_max, win, step, var = parent._calculate_params(N, freq_base,
                freq_max, hop, win_shape)
        transform = self.transform
        for samples in self.audio.walk(win, step, start, end, 
                join_channels=True):
            v = transform(samples, n, k_max, norm=False, pre_var=var)
            pw = (v.real**2 + v.imag**2) / var.WL
            yield pw

class Y2(Y1):
    
    @staticmethod
    def g(vector, gamma=10):
        if not gamma or gamma >= inf:
            return vector / 2
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


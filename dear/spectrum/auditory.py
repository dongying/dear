#-*- coding: utf-8 -*-

from _base import *
from cqt import CNTPowerSpectrum, A0, A1, A2, C8, A8
import numpy as np 
import scipy.signal as sig

inf = float('inf')


class GammatoneSpectrum(SpectrumBase):

    @staticmethod
    def erb_space(N, freq_base, freq_max):
        EarQ = 9.26449
        minBW = 24.7
        qw = EarQ * minBW
        arr = np.arange(1., N+1, 1., dtype=np.double)
        f_cents = np.exp(arr*(np.log(freq_base+qw)-np.log(freq_max+qw))/N)\
                * (freq_max+qw) - qw
        return f_cents 

    @staticmethod
    def make_erb_filter_coeffiences(samplerate, N=64, freq_base=A2, freq_max=C8):
        fc = GammatoneSpectrum.erb_space(N, freq_base, freq_max)
        #
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        sqrt = np.sqrt
        #
        EarQ = 9.26449
        minBW = 24.7
        order = 1.
        #ERB = ((fc/EarQ)**order + minBW**order) ** (1./order)
        ERB = 0.108*fc + minBW
        BW = 1.019*2*pi*ERB
        #
        T = 1./samplerate
        fct = np.flipud(fc*T)
        bwt = np.flipud(BW*T)
        a0 = T
        a2 = 0
        b0 = 1
        b1 = -2*cos(2*pi*fct) / exp(bwt)
        b2 = exp(-2*bwt)
        a11 = -(2*T*cos(2*pi*fct)/exp(bwt)+2*T*sqrt(3+2**1.5)*sin(2*pi*fct)/exp(bwt)) / 2
        a12 = -(2*T*cos(2*pi*fct)/exp(bwt)-2*T*sqrt(3+2**1.5)*sin(2*pi*fct)/exp(bwt)) / 2
        a13 = -(2*T*cos(2*pi*fct)/exp(bwt)+2*T*sqrt(3-2**1.5)*sin(2*pi*fct)/exp(bwt)) / 2
        a14 = -(2*T*cos(2*pi*fct)/exp(bwt)-2*T*sqrt(3-2**1.5)*sin(2*pi*fct)/exp(bwt)) / 2
        gain = np.abs(
              (-2*exp(4j*pi*fct)*T + 2*exp(2j*pi*fct-bwt)*T*(cos(2*pi*fct) - sqrt(3-2**1.5)*sin(2*pi*fct)))\
            * (-2*exp(4j*pi*fct)*T + 2*exp(2j*pi*fct-bwt)*T*(cos(2*pi*fct) + sqrt(3-2**1.5)*sin(2*pi*fct)))\
            * (-2*exp(4j*pi*fct)*T + 2*exp(2j*pi*fct-bwt)*T*(cos(2*pi*fct) - sqrt(3+2**1.5)*sin(2*pi*fct)))\
            * (-2*exp(4j*pi*fct)*T + 2*exp(2j*pi*fct-bwt)*T*(cos(2*pi*fct) + sqrt(3+2**1.5)*sin(2*pi*fct)))\
            / (-2/exp(2*bwt) - 2*exp(4j*pi*fct) + 2*(1+exp(4j*pi*fct))/exp(bwt)) ** 4)
        #
        #return a0,a11,a12,a13,a14,a2,b0,b1,b2,gain
        coeffies = np.zeros((5, N, 3))
        coeffies[0][...,0], coeffies[0][...,1], coeffies[0][...,2] = b0, b1, b2
        coeffies[1:,:,0], coeffies[1:,:,2] = a0, a2
        for i,p1 in enumerate([a11,a12,a13,a14]):
            coeffies[i+1,:,1] = p1
        coeffies[1,:,[0,1,2]] /= gain
        return coeffies

    @staticmethod
    def filter(samples, coefficiences, zi=None):
        n_filters = coefficiences.shape[1]
        if zi is None:
            zi = np.zeros((4, n_filters, coefficiences.shape[-1]-1))
        mtx = numpy.zeros((len(samples), n_filters))
        #
        for i in range(n_filters):
            b = coefficiences[0,i]
            a1, a2, a3, a4 = coefficiences[1:,i]
            s1, zi[0,i] = sig.lfilter(a1,b,samples,zi=zi[0,i])
            s2, zi[1,i] = sig.lfilter(a2,b,s1,zi=zi[1,i])
            s3, zi[2,i] = sig.lfilter(a3,b,s2,zi=zi[2,i])
            s4, zi[3,i] = sig.lfilter(a4,b,s3,zi=zi[3,i])
            #print s1,s2,s3,s4
            mtx[...,i] = s4
        return mtx, zi

    def walk(self, N=64, freq_base=A2, freq_max=C8,  start=0, end=None, each=True, combine=False, twin=0.025, thop=0.010):
        ''''''
        N = int(N)
        assert N > 0
        #
        samplerate = self.audio.samplerate
        assert 1 <= freq_base <= freq_max <= samplerate/2.0
        #
        step = int(2**np.ceil(np.log2(float(samplerate) / N)))
        win = step
        assert 0 < step <= win
        #
        coeffies = self.make_erb_filter_coeffiences(samplerate, N, freq_base, freq_max)
        zi = None
        if not combine:
            for samples in self.audio.walk(win, step, start, end, join_channels=True):
                y, zi = self.filter(samples, coeffies, zi)
                if each:
                    for frame in y: yield frame
                else:
                    yield y
        else:
            cstep = int(np.ceil(thop*samplerate))
            cwin = int(np.ceil(twin*samplerate))
            assert 0 < thop <= twin
            assert 0 < cstep <= cwin
            Y = np.zeros((0,N))
            for samples in self.audio.walk(win, step, start, end, join_channels=True):
                y, zi = self.filter(samples, coeffies, zi)
                Y = np.append(Y, y, 0)
                while Y.shape[0] >= cwin:
                    wf, Y = Y[:cwin], Y[cstep:]
                    yield np.sqrt(np.mean(np.square(wf), 0))
            if Y.shape[0] > 0:
                yield np.sqrt(np.mean(np.square(Y), 0))


class Y1(GammatoneSpectrum):
    pass


class Y2(Y1):
    
    @staticmethod
    def g(y, gamma=10):
        if not gamma or gamma >= inf:
            return 0.5
        return 1./(1+np.exp(-gamma*y)) - 0.5

    def walk(self, N=64, freq_base=A2, freq_max=C8,  start=0, end=None, each=True, combine=False, twin=0.025, thop=0.010, gamma=10, cof=4500):
        ''''''
        samplerate = self.audio.samplerate
        level = 6
        vpre = np.zeros((1,N))
        ncof = min(0.99, float(cof)*2/samplerate)
        b,a = sig.butter(6, ncof)
        zi = np.zeros((N,level))
        if not combine:
            for y in super(Y2, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False):
                yg = self.g( 
                        np.diff(np.append(vpre,y,0), 1, 0),
                        gamma)
                ygt, zi = sig.lfilter(b, a, yg.T, zi=zi)
                if each:
                    for v in ygt.T: yield v
                else:
                    yield ygt.T
        else:
            cstep = int(np.ceil(thop*samplerate))
            cwin = int(np.ceil(twin*samplerate))
            assert 0 < thop <= twin
            assert 0 < cstep <= cwin
            Y = np.zeros((0,N))
            for y in super(Y2, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False):
                yg = self.g( 
                        np.diff(np.append(vpre,y,0), 1, 0),
                        gamma)
                vpre = y[-1:]
                ygt, zi = sig.lfilter(b, a, yg.T, zi=zi)
                Y = np.append(Y, ygt.T, 0)
                while Y.shape[0] >= cwin:
                    wf, Y = Y[:cwin], Y[cstep:]
                    yield np.sqrt(np.mean(np.square(wf), 0))
            if Y.shape[0] > 0:
                yield np.sqrt(np.mean(np.square(Y), 0))


class Y3(Y2):
    def walk(self, N=64, freq_base=A2, freq_max=C8,  start=0, end=None, each=True, combine=False, twin=0.025, thop=0.010, gamma=10, cof=4500):
        if not combine:
            for y in super(Y3, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False,gamma=gamma,cof=cof):
                y[...,1:] = np.diff(y,1,1)
                if each:
                    for v in y: yield v
                else:
                    yield y
        else:
            samplerate = self.audio.samplerate
            cstep = int(np.ceil(thop*samplerate))
            cwin = int(np.ceil(twin*samplerate))
            assert 0 < thop <= twin
            assert 0 < cstep <= cwin
            Y = np.zeros((0,N))
            for y in super(Y3, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False,gamma=gamma,cof=cof):
                y[...,1:] = np.diff(y,1,1)
                Y = np.append(Y, y, 0)
                while Y.shape[0] >= cwin:
                    wf, Y = Y[:cwin], Y[cstep:]
                    yield np.sqrt(np.mean(np.square(wf), 0))
            if Y.shape[0] > 0:
                yield np.sqrt(np.mean(np.square(Y), 0))


class Y4(Y3):
    def walk(self, N=64, freq_base=A2, freq_max=C8,  start=0, end=None, each=True, combine=False, twin=0.025, thop=0.010, gamma=10, cof=4500):
        if not combine:
            for y in super(Y4, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False,gamma=gamma,cof=cof):
                y = np.maximum(0, y)
                if each:
                    for v in y: yield v
                else:
                    yield y
        else:
            samplerate = self.audio.samplerate
            cstep = int(np.ceil(thop*samplerate))
            cwin = int(np.ceil(twin*samplerate))
            assert 0 < thop <= twin
            assert 0 < cstep <= cwin
            Y = np.zeros((0,N))
            for y in super(Y4, self).walk(N,freq_base,freq_max,start,end,each=False,combine=False,gamma=gamma,cof=cof):
                y = np.maximum(0, y)
                Y = np.append(Y, y, 0)
                while Y.shape[0] >= cwin:
                    wf, Y = Y[:cwin], Y[cstep:]
                    yield np.sqrt(np.mean(np.square(wf), 0))
            if Y.shape[0] > 0:
                yield np.sqrt(np.mean(np.square(Y), 0))


class Y5(Y4):
    pass


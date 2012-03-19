#-*- coding: utf-8 -*-

from _base import *
import numpy


A0 = 27.5
A1 = 55.0
A2 = 110.
C8 = 4186.01
A8 = 7040.0

class Spectrum(SpectrumBase):
    '''Spectrum of Constant-Q Transform'''

    @staticmethod
    def pre_calculate(Q, k_max, win, win_shape):
        var = {}
        #
        t = 1 + 1/float(Q)
        WL = [max(2, int(round(win / t**k))) for k in xrange(k_max)]
        var['WL'] = WL
        PRE = []
        for wl in WL:
            arr = 2.*numpy.pi * numpy.arange(wl) / wl
            PRE.append(
                win_shape(wl) * (numpy.cos(arr*Q) - numpy.sin(arr*Q)*1j)
            )
        var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform(samples, Q, k_max=None, win_shape=numpy.hamming,
            pre_var=None):
        if not pre_var:
            if not k_max:
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) \
                        / numpy.log2(float(Q+1)/Q))
            pre_var = Spectrum.pre_calculate(Q,k_max,len(samples),win_shape)
        frame = numpy.array(
            [numpy.sum(samples[:wl] * pre) / wl \
                for wl,pre in zip(pre_var.WL, pre_var.PRE)])
        return frame

    def walk(self, Q, freq_base=A0, freq_max=C8, hop=0.02, start=0, end=None,
            join_channels=True, win_shape=numpy.hamming):
        ''''''
        #
        Q = int(Q)
        assert Q > 1
        #
        samplerate = self.audio.samplerate
        if not freq_max: freq_max = samplerate/2.0
        assert 1 <= freq_base <= freq_max <= samplerate/2.0
        #
        step = int(samplerate * hop)
        win = int(round(Q * float(samplerate) / freq_base))
        assert 0 < step <= win 
        #
        k_max = int(numpy.log2(float(freq_max)/freq_base) \
                / numpy.log2(float(Q+1)/Q))
        #
        var = self.pre_calculate(Q, k_max, win, win_shape)
        print len(var.WL), var.WL
        fqs = []
        for wl in var.WL:
            fqs.append("%.2f" % (float(samplerate) / wl * Q))
        print fqs
        transform = self.transform
        #
        for samples in self.audio.walk(win, step, start, end, join_channels):
            if join_channels:
                yield transform(samples, Q, k_max, pre_var=var)
            else:
                yield [transform(ch,Q,k_max,pre_var=var) \
                        for ch in samples]


class CQTPowerSpectrum(Spectrum):
    @staticmethod
    def transform(samples, Q, k_max=None, win_shape=numpy.hamming,
            pre_var=None):
        if not pre_var:
            if not k_max:
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) \
                        / numpy.log2(float(Q+1)/Q))
            pre_var = CQTPowerSpectrum.pre_calculate(Q,k_max,len(samples),win_shape)
        frame = numpy.array(
            [numpy.sum(samples[:wl] * pre) \
                for wl,pre in zip(pre_var.WL, pre_var.PRE)])
        return (frame.real**2 + frame.imag**2) / pre_var.WL



class CNTSpectrum(SpectrumBase):
    MIN_Q = 8

    @staticmethod
    def pre_calculate(N, k_max, win, win_shape, sr=None, resize=True):
        var = {}
        #
        Q_f = 1. / (2.**(1./N) - 1)
        Q = max(2, int(round(Q_f)))
        QV = numpy.array([Q] * (k_max))
        WL = numpy.array(
                [win/2.**(float(k)/N) for k in xrange(k_max)])
        if resize:
            if sr is None:
                sr = win
            tmp_q = CNTSpectrum.MIN_Q
            for i,wl in enumerate(WL):
                delta_wl = float(wl) / Q_f
                tmp_wl = int(round(delta_wl * tmp_q))
                tmp_wl_next = int(round(delta_wl * (tmp_q+1)))
                while tmp_wl < sr/10. and (tmp_wl_next) <= win:
                    tmp_q += 1
                    tmp_wl = tmp_wl_next
                    tmp_wl_next = int(round(delta_wl * (tmp_q+1)))
                if tmp_wl <= win:
                    QV[i] = tmp_q
                    WL[i] = tmp_wl
        WL = numpy.array(numpy.round(WL), int)
        var['WL'] = WL
        var['QV'] = QV
        #
        PRE = []
        for q, wl in zip(QV,WL):
            arr = 2.*numpy.pi * numpy.arange(wl) / wl
            PRE.append(
                win_shape(wl) * (numpy.cos(arr*q) - numpy.sin(arr*q)*1j)
            )
        var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform(samples, N, k_max=None, win_shape=numpy.hanning, 
            norm=True, pre_var=None):
        if not pre_var:
            if not k_max:
                N = int(N)
                assert N > 1 
                Q = 1. / (2.**(1./N) - 1)
                Q = max(2, int(round(Q)))
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) * N)
            pre_var = Spectrum.pre_calculate(N,k_max,len(samples),win_shape)
        frame = numpy.array(
            [numpy.sum(samples[:wl] * pre)\
                for wl,pre in zip(pre_var.WL, pre_var.PRE)])
        if norm:
            return frame / pre_var.WL
        return frame

    def _calculate_params(self, N, freq_base=A0, freq_max=A8, hop=0.02,
            win_shape=numpy.hanning, resize_win=True):
        N = int(N)
        assert N > 1
        Q = 1. / (2.**(1./N) - 1)
        Q = max(2, int(round(Q)))
        #
        samplerate = self.audio.samplerate
        if not freq_max: freq_max = samplerate/2.0
        assert 1 <= freq_base <= freq_max <= samplerate/2.0
        #
        step = int(samplerate * hop)
        win_f = Q * float(samplerate) / freq_base
        win = int(round(win_f))
        assert 0 < step <= win 
        #
        k_max = int(numpy.log2(float(freq_max)/freq_base) * N)
        var = self.pre_calculate(N, k_max, win_f, win_shape, samplerate, 
                resize=resize_win)
        print var.WL.shape, var.WL
        print var.QV
        win = var.WL.max()
        fqs = []
        for q,wl in zip(var.QV,var.WL):
            fqs.append("%.2f" % (float(self.audio.samplerate) / wl * q))
        print fqs
        #
        return N, k_max, win, step, var

    def walk(self, N, freq_base=A0, freq_max=C8, hop=0.02, start=0, end=None,
            join_channels=True, win_shape=numpy.hanning, resize_win=True):
        ''''''
        n, k_max, win, step, var = self._calculate_params(N, freq_base,
                freq_max, hop, win_shape, resize_win)
        transform = self.transform
        for samples in self.audio.walk(win, step, start, end, join_channels):
            if join_channels:
                yield transform(samples, n, k_max, pre_var=var)
            else:
                yield [transform(ch,n,k_max,pre_var=var) \
                        for ch in samples]


class CNTPowerSpectrum(CNTSpectrum):

    @staticmethod
    def transform(samples, N, k_max=None, win_shape=numpy.hanning, 
            pre_var=None):
        if not pre_var:
            if not k_max:
                N = int(N)
                assert N > 1 
                Q = 1. / (2.**(1./N) - 1)
                Q = max(2, int(round(Q)))
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) * N)
            pre_var = Spectrum.pre_calculate(N,k_max,len(samples),win_shape)
        frame = numpy.array(
            [numpy.sum(samples[:wl] * pre)\
                for wl,pre in zip(pre_var.WL, pre_var.PRE)])
        return (frame.real**2 + frame.imag**2) / pre_var.WL



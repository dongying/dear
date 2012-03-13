#-*- coding: utf-8 -*-

import numpy as np
from dear.spectrum import DFTSpectrum


class MFCCs(DFTSpectrum):

    @staticmethod
    def freq2mel(freq):
        return 1000 * np.log(1+freq/700.0) / np.log(1+1000.0/700.0)

    @staticmethod
    def mel2freq(mel):
        return 700 * (np.exp(mel/1000.*np.log(1+1000.0/700.0)) - 1)

    @staticmethod
    def pre_calculate(N, win, freq_min, freq_max, nqfreq):
        bins = int(win)/2
        #
        mel_min, mel_max = MFCCs.freq2mel(freq_min), MFCCs.freq2mel(freq_max)
        mel_centrs = np.arange(N+2) * (mel_max-mel_min) / (N+1) + mel_min
        bin_centrs = np.round(MFCCs.mel2freq(mel_centrs) / nqfreq * bins)
        #
        filters = np.zeros((N, bins))
        for i in xrange(N):
            st, ct, ed = bin_centrs[i:i+3]
            assert st < ct < ed
            #print st, ct, ed
            up = (np.arange(st,ct) - st) / (ct-st)
            down = (ed - np.arange(ct,ed)) / (ed-ct)
            filters[i,int(st):int(ct)] = up
            filters[i,int(ct):int(ed)] = down
        #
        dc = np.ones((N,N)) * np.arange(N,dtype=np.double)
        dcmt = np.pi * (0.5+dc) / N
        dcmt = (1.0 / np.sqrt(N/2.0)) * np.cos(dc.T * dcmt)
        dcmt[0] /= np.sqrt(2.0)
        #
        var = {
            'WL': win,
            'Triangles': filters,
            'DCTMatrix': dcmt
        }
        return type('variables', (object,), var)

    @staticmethod
    def transform(frame, var):
        f = np.dot(np.abs(frame[:-1])*var.WL, var.Triangles.T)
        f = np.dot(np.log(f), var.DCTMatrix)
        return f

    def walk(self, N=20, freq_min=0, freq_max=7040., win=2048, step=1024, start=0, end=None, win_shape=np.hanning):
        samplerate = self.audio.samplerate
        nqfreq = samplerate / 2.
        #
        N = int(N)
        assert 0 < N < int(win)/2
        #
        freq_max = (freq_max is None) and nqfreq or freq_max
        assert 0<= freq_min < freq_max <= nqfreq
        #
        var = MFCCs.pre_calculate(N, win, freq_min, freq_max, nqfreq)
        for samples in super(MFCCs, self).walk(win, step, start, end, join_channels=True, win_shape=win_shape):
            frame = MFCCs.transform(samples, var)
            yield frame


#-*- coding: utf-8 -*-


class SpectrumBase(object):
    '''Base type of spectrums'''

    def __init__(self, audio):
        self.audio = audio

    def walk(self, win, step, start, end, join_channels):
        '''Generator of spectrum frames'''
        raise NotImplementedError


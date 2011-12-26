#-*- coding: utf-8 -*-

import os


class AudioBase(object):
    ''''''

    def __init__(self, path):
        ''''''
        self._path = path
        if not os.path.isfile(path):
            raise ValueError("`%s' is not a file." % path)

    @property
    def samplerate(self):
        '''Get sample rate.'''
        raise NotImplementedError

    @property
    def channels(self):
        '''Get number of channels.'''
        raise NotImplementedError

    @property
    def duration(self):
        '''Get duration in seconds.'''
        raise NotImplementedError

    def __len__(self):
        return self.duration

    def walk(self, win=None, step=None, start=None, end=None, channels=None, join=None):
        '''Generator of samples by window.'''
        raise NotImplementedError


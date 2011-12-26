#-*- coding: utf-8 -*-

from _io_base import AudioBase
import mad


support_formats = ['mp3']


class Decoder(AudioBase):
    ''''''

    def __init__(self, path):
        ''''''
        super(Decoder, self).__init__(path)
        self.fh = mad.MadFile(path)

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


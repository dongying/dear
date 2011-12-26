#-*- coding: utf-8 -*-

from _io_base import AudioBase
import audioread


support_formats = ['mp3','wav','aiff']


class Decoder(AudioBase):
    ''''''

    def __init__(self, path):
        ''''''
        super(Decoder, self).__init__(path)
        self.fh = audioread.audio_open(path)

    @property
    def samplerate(self):
        return self.fh.samplerate

    @property
    def channels(self):
        return self.fh.channels

    @property
    def duration(self):
        return self.fh.duration

    def __len__(self):
        return self.duration

    def walk(self, win=None, step=None, start=None, end=None, channels=None,
            join=None):
        


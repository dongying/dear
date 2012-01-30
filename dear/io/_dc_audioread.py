#-*- coding: utf-8 -*-

from _io_base import AudioBase
import audioread
import struct, numpy, itertools


support_formats = ['mp3','wav','aiff']


class Audio(AudioBase):
    ''''''

    def __init__(self, path):
        super(Audio, self).__init__(path)
        fh = audioread.audio_open(path)
        self._samplerate = fh.samplerate
        self._channels = fh.channels
        self._duration = fh.duration
        fh.close()

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def channels(self):
        return self._channels

    @property
    def duration(self):
        return self._duration

    def __len__(self):
        return self._duration

    _max_pulse_value = 2**15

    def _pcm_bin2num(self, bins, channels, join=False):
        pcms = numpy.array(
                struct.unpack('<'+'h'*(len(bins)/2), bins), 
                float)
        pcms = pcms / self._max_pulse_value
        if channels <= 1:
            return join and pcms or [pcms]
        chs = []
        for i in range(channels):
            chs.append(
                numpy.array(
                    [pcms[i] for i in xrange(i, len(pcms), channels)]))
        if not join:
            return chs
        return sum(chs) / channels

    def walk(self, win=1024, step=None, start=0, end=None,
            join_channels=False):
        if not step:
            step = win
        if not end:
            end = float('inf')
        assert win >= step > 0
        assert start > 0
        if start >= self._duration or end <= start:
            raise StopIteration
        #
        fh = audioread.audio_open(self._path)
        total = fh.samplerate * fh.duration / 1024.0
        start_point = int(float(start) / fh.duration * total)
        end_point = int(float(end) / fh.duration * total)
        win_len = int(win) * fh.channels * 2
        step_len = int(step) * fh.channels * 2
        #
        win_buf = ''
        for p, buf in enumerate(fh):
            if p < start_point:
                continue
            if p >= end_point:
                break
            win_buf += buf
            while len(win_buf) > win_len:
                nbuf, win_buf = win_buf[:win_len], win_buf[step_len:]
                yield self._pcm_bin2num(nbuf, fh.channels, join_channels)
        #
        win_buf += '\0' * (win_len - len(win_buf))
        yield self._pcm_bin2num(win_buf, fh.channels, join_channels)
        fh.close()


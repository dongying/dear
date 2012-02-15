#-*- coding: utf-8 -*-

import os, struct, numpy

dB_MIN = -80.
PWI_THS = 60.
PW_MIN = PWI_THS * (10.** (dB_MIN / 20.))


class SpectrumBase(object):
    '''Base type of spectrums'''

    def __init__(self, audio):
        self.audio = audio

    def walk(self, win, step, start, end, join_channels):
        '''Generator of spectrum frames'''
        raise NotImplementedError


class SpectrogramFile(object):
    '''Read or Write spectrogram into file'''

    def __init__(self, path, mode='r'):
        assert mode in ('r','w')
        self._mode = mode
        if self._mode == 'r' \
                and not os.path.isfile(path):
            raise ValueError("`%s' is not a file." % path)
        self._path = path
        self._fh = open(self._path, mode+'b')

    def _read_header(self):
        '''
        Little-endian
        |... 4 bytes int ...|... 4 bytes int ...|
        |   frames count    | dimensions count  |
        '''
        self._fh.seek(0)
        buf = self._fh.read(4*2)
        fc, dc = struct.unpack("<II", buf)
        return fc, dc

    def _write_header(self, fc, dc):
        self._fh.seek(0)
        self._fh.write(struct.pack("<II",int(fc), int(dc)))

    def _read_frame(self, dc):
        buf = self._fh.read(dc * 8)
        if len(buf) < 8:
            return None
        return numpy.array(struct.unpack('<'+'d'*dc, buf))

    def _write_frame(self, vector):
        buf = struct.pack('<'+'d'*len(vector), *vector)
        self._fh.write(buf)

    def close(self):
        self._fh.close()

    def walk(self, offset=0, limit=None):
        fc, dc = self._read_header()
        offset = int(offset)
        limit = limit is None and fc or int(limit)
        assert 0 <= offset < fc
        end = offset + limit
        end = end > fc and fc or end
        if offset > 0:
            self._fh.seek(offset * 8 * dc, os.SEEK_CUR)
        for idx in xrange(offset, end):
            vector = self._read_frame(dc)
            if vector is None:
                raise StopIteration
            yield vector

    def dump(self, spectrum_iter):
        self._write_header(0,0)
        fc, dc = 0, None
        for vector in spectrum_iter:
            self._write_frame(vector)
            fc += 1
            if dc is None:
                dc = len(vector)
        self._write_header(fc, dc)


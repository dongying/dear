#-*- coding: utf-8 -*-

import numpy as np
from dear.spectrum import DFTPowerSpectrum


def crop_mmag(audio, start=0, end=None, length=30):
    if end is None: end = audio.duration
    end = (end <= audio.duration) and end or audio.duration
    assert 0 <= start < end
    assert 0 < length
    if end - start <= length:
        return start, end
    #
    win = 512
    time_hop = float(win)/audio.samplerate
    max_mag = 0
    cur_mag = 0
    mag_arr = [0]
    now = start
    new_end = start + length
    for samples in audio.walk(win=win, step=win, start=start, end=end, join_channels=True):
        now += time_hop
        mag = np.sum(np.abs(samples))
        mag_arr.append(mag)
        if now > new_end:
            cur_mag += mag - mag_arr.pop(0)
            if cur_mag > max_mag:
                new_end = now
                max_mag = cur_mag
        else:
            cur_mag += mag
            max_mag = cur_mag
        #print start, now, cur_mag, new_end, max_mag
    #
    return int(new_end - length), int(new_end)
    #gap = length * (1 - 0.618)
    #new_start = new_end - length
    #if new_start - start >= gap:
    #    new_start -= gap
    #    return int(new_start), int(new_start + length)
    #return int(start), int(start + length)


def crop_mpwg(audio, start=0, end=None, length=30):
    if end is None: end = audio.duration
    end = (end <= audio.duration) and end or audio.duration
    assert 0 <= start < end
    assert 0 < length
    if end - start <= length:
        return start, end
    #
    win = 512
    time_hop = float(win)/audio.samplerate
    max_mag = 0
    cur_mag = 0
    mag_arr = [0]
    now = start
    new_end = start + length
    spec = DFTPowerSpectrum(audio)
    for frame in spec.walk(win=win, step=win, start=start, end=end, join_channels=True):
        now += time_hop
        mag = np.sum(frame)
        mag_arr.append(mag)
        if now > new_end:
            cur_mag += mag - mag_arr.pop(0)
            if cur_mag > max_mag:
                new_end = now
                max_mag = cur_mag
        else:
            cur_mag += mag
            max_mag = cur_mag
    return int(new_end - length), int(new_end)


CROP_ALGORITHMS = {
    'mmag': crop_mmag,
    'mpwg': crop_mpwg
}


if __name__ == '__main__':
    import getopt, sys, traceback, os, subprocess

    def exit_with_usage():
        print """Usage: $ python -m dear.crop <options>
Options:
     -i     input path
     -o     output path
    [-l]    length of clip, default 30 seconds.
    [-s]    start time in second, defaute 0
    [-t]    end time, default is duration of song
    [-a]    algorithm, could be one of ('mmag','mpwg'), default 'mmag'
    [-r]    samplerate, default 22050
    [-b]    bitrate, default 64k(bit)
    [-d]    dim seconds of begining and ending, default 4
"""
        exit()

    def print_exc():
        print "-"*72
        traceback.print_exc()
        print "-"*72


    if len(sys.argv) < 4:
        exit_with_usage()
    try:
        inputf = None
        output = None
        length = 30
        start = 0
        end = None
        algorithm = 'mmag'
        samplerate = 22050
        bitrate = '64k'
        do_crop = True
        dim = 4
        #
        opts, args = getopt.getopt(sys.argv[1:], "i:o:l:s:t:a:r:b:d:")
        for o, a in opts:
            if o == '-i':
                inputf = a
            elif o == '-o':
                output = a
            elif o == '-l':
                length = int(a)
            elif o == '-s':
                start = int(a)
            elif o == '-t':
                end = int(a)
            elif o == '-a':
                algorithm = a
            elif o == '-r':
                samplerate = int(a)
            elif o == '-b':
                bitrate = a
            elif o == '-d':
                dim = int(a)
        assert os.path.isfile(inputf)
        assert output is not None
        assert algorithm in CROP_ALGORITHMS
        assert 0 < length
        assert (end is None) or 0 <= start < end
        assert 0 < samplerate
        assert 0 < bitrate
    except Exception as ex:
        print_exc()
        exit_with_usage()
    if len(args) != 0:
        exit_with_usage()

    func = CROP_ALGORITHMS.get(algorithm)

    import dear.io as io
    decoder = io.get_decoder(name='audioread')
    audio = decoder.Audio(inputf)
    print "SampleRate: %d Hz\nChannel(s): %d\nDuration: %d sec"\
            % (audio.samplerate, audio.channels, audio.duration)

    if start >= audio.duration:
        print "[error] Start time is beyond song duration. Not cropping."
    else:
        start, end = func(audio, start, end, length)
    duration = end - start
    print start, end, duration

    tmpfile = output + '.crop.tmp.wav'
    cmd = ['ffmpeg','-i',inputf,'-ss',str(start),'-t',str(duration),
            '-ac','1','-ar',str(samplerate),'-ab',bitrate,
            tmpfile]
    p = subprocess.Popen(cmd)
    p.wait()

    if dim > 0 and duration > 2*dim:
        cmd = ['sox',tmpfile,output,'fade','p',str(dim),
                str(duration-1),str(dim)]
        p = subprocess.Popen(cmd)
        p.wait()
    else:
        p = subprocess.Popen(['mv', tmpfile, output])
        p.wait()
    os.remove(tmpfile)


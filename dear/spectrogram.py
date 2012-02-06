#-*- coding: utf-8 -*-

import numpy
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage
import matplotlib.colors as colo

from dear.spectrum import cqt, dft


def plot_spectrogram(spec, Xd=(0,1), Yd=(0,1)):
    #
    x_min, x_max = Xd
    y_min, y_max = Yd
    #
    fig = plt.figure()
    nf = len(spec)
    for ch, data in enumerate(spec):
        #print ch, data.shape
        x = numpy.linspace(x_min, x_max, data.shape[0])
        y = numpy.linspace(y_min, y_max, data.shape[1])
        #print x[0],x[-1],y[0],y[-1]
        ax = fig.add_subplot(nf*100+11+ch)
        im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.gray_r,
                norm=colo.LogNorm(vmin=.00001))
        im.set_data(x, y, data.T)
        ax.images.append(im)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title('Channel %d' % ch)
        #ax.set_xlabel('timeline')
        ax.set_ylabel('frequency')
        print 'Statistics: max<%.3f> min<%.3f> mean<%.3f> median<%.3f>' % (data.max(), data.min(), data.mean(), numpy.median(data))
    #
    plt.show()


if __name__ == '__main__':
    import getopt, sys

    def exit_with_usage():
        print """Usage: $ python -m dear.spectrogram <options> /path/to/song

Option:
    [-s]    start time in second, default 0
    [-t]    end time, default is duration of song
    [-g]      type of spectrogram, default dft:

        dft --- Discrete Fourier Transform
            [-w]    window size, default 2048
            [-h]    step size, default 512

        cqt --- Constant-Q Transform
            [-q]    Q, default 34
            [-h]    hop in second, default 0.02

        cnt --- Constant-N transform
            [-n]    N, default 24
            [-h]    hop in second, default 0.02
"""
        exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:s:t:h:w:q:n:")
    except getopt.GetoptError as ex:
        print ex
        exit_with_usage()
    if len(args) != 1:
        #print args
        exit_with_usage()

    import dear.io as io
    decoder = io.get_decoder(name='audioread')
    audio = decoder.Audio(args[0])
    print "SampleRate: %d Hz\nChannel(s): %d\nDuration: %d sec"\
            % (audio.samplerate, audio.channels, audio.duration)

    graph = 'dft'
    st = 0
    to = None

    for o, a in opts:
        if o == '-s':
            st = float(a)
        elif o == '-t':
            to = float(a)
        elif o == '-g':
            graph = a.lower()
            assert graph in ('dft','cqt','cnt')

    if graph == 'dft':
        win = 2048
        hop = 512
        for o, a in opts:
            if o == '-w':
                win = int(a)
            elif o == '-h':
                hop = int(a)
        spec = [[]]
        gram = dft.Spectrum(audio)
        for freqs in gram.walk(win, hop, start=st, end=to, join_channels=True):
            spec[0].append(abs(freqs))
    #
    elif graph == 'cqt':
        Q = 34
        hop = 0.02
        for o, a in opts:
            if o == '-q':
                Q = int(a)
            elif o == '-h':
                hop = float(a)
        spec = [[]]
        gram = cqt.Spectrum(audio)
        for freqs in gram.walk(Q=Q, hop=hop, start=st, end=to, join_channels=True):
            spec[0].append(abs(freqs))
    #
    elif graph == 'cnt':
        N = 24
        hop = 0.02
        for o, a in opts:
            if o == '-n':
                N = int(a)
            elif o == '-h':
                hop = float(a)
        spec = [[]]
        gram = cqt.CNTSpectrum(audio)
        for freqs in gram.walk(N=N, hop=hop, start=st, end=to, join_channels=True):
            spec[0].append(abs(freqs))

    if to is None or to > audio.duration:
        to = audio.duration

    plot_spectrogram(numpy.array(spec), (st,to), (0.,1.))


#-*- coding: utf-8 -*-

from _base import SpectrumBase
import numpy


class Spectrum(SpectrumBase):
    '''Spectrum of Discrete Fourier Transform'''
    
    @staticmethod
    def pre_calculate(win, win_shape):
        var = {
            'WL': win,
            'W': win_shape(win)}
        #
        PRE = []
        arr = 2. * numpy.pi * numpy.arange(win) / win
        for k in xrange(win/2+1):
            PRE.append(
                var['W'] * (numpy.cos(arr*k) - numpy.sin(arr*k)*1j) 
            )
        var['PRE'] = PRE
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform(samples, win_shape=numpy.hamming, pre_var=None):
        if not pre_var:
            pre_var = Spectrum.pre_calculate(len(samples), win_shape)
        frame = numpy.array(
                [numpy.sum(samples * pre) for pre in pre_var.PRE])
        return frame / pre_var.WL

    def walk(self, win=1024, step=512, start=0, end=None, join_channels=True,
            win_shape=numpy.hamming):
        var = self.pre_calculate(win, win_shape)
        #
        for samples in self.audio.walk(win, step, start, end, join_channels):
            if join_channels:
                yield self.transform(samples, pre_var=var)
            else: yield [self.transform(ch, pre_var=var) for ch in samples]


def plot_spectrogram(spec, Xd=(0,1), Yd=(0,1)):
    import matplotlib
    #matplotlib.use('GTKAgg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.image import NonUniformImage
    import matplotlib.colors as colo
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
        print "Usage: $ python cqt.py [-s start_sec] [-t to_sec] [-w win] [-h hop] /path/to/song"
        exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:t:w:h:")
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

    st = 0
    to = None
    win = 1024
    hop = 512

    for o, a in opts:
        if o == '-s':
            st = float(a)
        elif o == '-t':
            to = float(a)
        elif o == '-w':
            win = int(a)
        elif o == '-h':
            hop = int(a)

    spec = [[]]
    gram = Spectrum(audio)
    for freqs in gram.walk(win, hop, start=st, end=to, join_channels=True):
        spec[0].append(abs(freqs))

    if to is None:
        to = audio.duration

    plot_spectrogram(numpy.array(spec), (st,to), (0.,1.))


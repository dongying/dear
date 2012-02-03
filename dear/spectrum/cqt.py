#-*- coding: utf-8 -*-

from _base import SpectrumBase
import numpy


A0 = 27.5
A1 = 55.0


class Spectrum(SpectrumBase):
    '''Spectrum of Constant-Q Transform'''

    @staticmethod
    def pre_calculate(Q, k_max, win, win_shape, pre=True):
        var = {}
        #
        t = 1 + 1/float(Q)
        WL = [max(2, int(round(win / t**k))) for k in xrange(k_max+1)]
        var['WL'] = WL
        if pre:
            PRE = []
            for wl in WL:
                arr = 2.*numpy.pi * numpy.arange(wl) / wl
                PRE.append(
                    win_shape(wl) * (numpy.cos(arr*Q) - numpy.sin(arr*Q)*1j)
                )
            var['PRE'] = PRE
        else:
            var['W'] = [win_shape(wl) for wl in WL]
        #
        return type('variables', (object,), var)

    @staticmethod
    def transform_pre(samples, Q, k_max=None, win_shape=numpy.hamming,
            pre_var=None):
        if not pre_var:
            if not k_max:
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) \
                        / numpy.log2(float(Q+1)/Q))
            pre_var = Spectrum.pre_calculate(Q,k_max,len(samples),win_shape,
                    pre=True)
        frame = numpy.array(
            [numpy.sum(samples[:wl] * pre) / wl \
                for wl,pre in zip(pre_var.WL, pre_var.PRE)])
        return frame

    @staticmethod
    def transform(samples, Q, k_max=None, win_shape=numpy.hamming,
            pre_var=None):
        if not pre_var:
            if not k_max:
                assert 1 < Q <= len(samples) / 2
                k_max = int(numpy.log2(float(len(samples))/Q/2) \
                        / numpy.log2(float(Q+1)/Q))
            pre_var = Spectrum.pre_calculate(Q,k_max,len(samples),win_shape,
                    pre=False)
        frame = []
        for k, (w, wl) in enumerate(zip(pre_var.W, pre_var.WL)):
            f = numpy.fft.rfft(w * samples[:wl])
            frame.append(f[Q] / wl)
        return numpy.array(frame)

    def walk(self, Q, freq_base=A0, freq_max=None, hop=0.02, start=0, end=None,
            join_channels=True, win_shape=numpy.hamming, mpre=False):
        ''''''
        #
        Q = int(Q)
        assert Q > 1
        #
        samplerate = self.audio.samplerate
        if not freq_max: freq_max = samplerate/2.0
        assert 1 <= freq_base <= freq_max <= samplerate/2.0
        #
        step = int(samplerate * hop)
        win = int(round(Q * float(samplerate) / freq_base))
        assert 0 < step <= win 
        #
        k_max = int(numpy.log2(float(freq_max)/freq_base) \
                / numpy.log2(float(Q+1)/Q))
        #
        var = self.pre_calculate(Q, k_max, win, win_shape, mpre)
        transform = mpre and self.transform_pre or self.transform
        #
        for samples in self.audio.walk(win, step, start, end, join_channels):
            if join_channels:
                yield transform(samples, Q, k_max, pre_var=var)
            else:
                yield [transform(ch,Q,k_max,pre_var=var) \
                        for ch in samples]


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
        print "Usage: $ python cqt.py [-s start_sec] [-t to_sec] [-h hop_sec] /path/to/song"
        exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:t:h:")
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
    hop = 0.020

    for o, a in opts:
        if o == '-s':
            st = float(a)
        elif o == '-t':
            to = float(a)
        elif o == '-h':
            hop = float(a)

    spec = [[]]
    gram = Spectrum(audio)
    for freqs in gram.walk(Q=34, hop=hop, start=st, end=to, join_channels=True):
        spec[0].append(abs(freqs))

    if to is None:
        to = audio.duration

    plot_spectrogram(numpy.array(spec), (st,to), (0.,1.))


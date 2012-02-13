#-*- coding: utf-8 -*-

import numpy


Bark = z = lambda f: 13*numpy.arctan(0.00076*f) + 3.5*numpy.arctan((f/7500.)**2)
SF = lambda z: 15.81 + 7.5*(z+0.474) - 17.5*numpy.sqrt(1+(z+0.474)**2)


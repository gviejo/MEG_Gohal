#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

from optparse import OptionParser
import numpy as np
import cPickle as pickle
from fusion_4 import fusion_4
from mixture_4 import mixture_4
from mixture_5 import mixture_5
from pylab import *

sys.path.append("../../src")

from Sferes import pareto, EA

with open("../p_test_last_set.pickle", 'rb') as f:
	p_test = pickle.load(f)

model = mixture_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test['S125']['mixture'], sferes = True)

with open("../meg/S12.pickle", "rb") as f:
	data = pickle.load(f)

opt = EA(data, 'S12', model)                                

for i in xrange(opt.n_blocs):
# for i in xrange(2):
    opt.model.startBloc()
    for j in xrange(opt.n_trials):
    # for j in xrange(5):
        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
        opt.model.updateValue(opt.responses[i,j])

opt.fit[0] = float(np.sum(opt.model.value))
opt.alignToMedian()
opt.fit[1] = float(-opt.leastSquares())                        
print opt.fit[0], opt.fit[1]

p = opt.mean[1]

c = np.genfromtxt("../Ccall/c.txt")

plot(np.cumsum(p))
plot(np.cumsum(c))

show()
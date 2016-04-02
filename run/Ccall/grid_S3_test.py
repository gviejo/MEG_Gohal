#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import *
import os, sys
from multiprocessing import Pool
# fit only the choice

nb_param = 7
subject = 'S3'

fit = np.zeros(tuple([5]*nb_param))

best = 0.0
best_parameter = np.zeros(nb_param)

for i in xrange(100000000):
	param = np.random.rand(nb_param)
	value = os.popen("./main "+" ".join(param.astype(str))).readlines()
	value = float(value[0].split(" ")[0])
	if value > best:
		best = value
		best_parameter = param
		print i, best


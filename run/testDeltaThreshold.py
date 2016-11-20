#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""

"""

import sys, os

from optparse import OptionParser
import numpy as np

sys.path.append("../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from Selection import *
from Sferes import pareto
from matplotlib import *
from pylab import *
import pickle
import matplotlib.pyplot as plt
from Sferes import pareto, EA

sys.path.append("set_meg_models")
from fusion_1 import fusion_1
from mixture_1 import mixture_1
from fusion_2 import fusion_2
from mixture_2 import mixture_2
from selection_1 import selection_1
from fusion_4 import fusion_4
from mixture_4 import mixture_4

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'
def convertStimulus(state):
    return (state == 's1')*1+(state == 's2')*2 + (state == 's3')*3
def convertAction(action):
    return (action=='thumb')*1+(action=='fore')*2+(action=='midd')*3+(action=='ring')*4+(action=='little')*5        

fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y : (y - fitfunc(p, x))

def leastSquares(x, y):
    for i in xrange(len(x)):
        pinit = [1.0, -1.0]
        p = leastsq(errfunc, pinit, args = (x[i], y[i]), full_output = False)
        x[i] = fitfunc(p[0], x[i])
    return x    

# def center(x):
#     x = x-np.median(x)
#     x = x/(np.percentile(x, 75)-np.percentile(x, 25))
#     return x
def center(x, s, m):    
    x = x-timing[s][m][0]
    x = x/timing[s][m][1]
    return x

# -----------------------------------
# MODELS 
# -----------------------------------
models = dict({'fusion':
                    {   1:fusion_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        2:fusion_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        4:fusion_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True)},
                'mixture': 
                    {   1:mixture_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        2:mixture_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        4:mixture_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True)},
                'selection':
                    {1:selection_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1,"eta":0.0001}, 0.05, 10, 0.1, True)}
            })

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 48
nb_repeat = 1000
case = 'meg'

human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../PEPS_GoHaL/fMRI',39)}))
cats = CATS(nb_trials)

# ------------------------------------
# Parameter testing change here
# ------------------------------------
# with open("p_test_last_set_v1.pickle", 'rb') as f:
#     p_test = pickle.load(f)
# with open("timing_v1.pickle", 'rb') as f:
#     timing = pickle.load(f)
with open("p_test_last_set.pickle", 'rb') as f:
    p_test = pickle.load(f)
with open("timing.pickle", 'rb') as f:
    timing = pickle.load(f)



s = 'S54'
m = 'fusion'
model = models[m][4]
model.setAllParameters(p_test[s][m])
with open("meg/"+s[0:-1]+".pickle", "rb") as f:
    data2 = pickle.load(f)
opt = EA(data2, s, model)                                
delta = np.zeros((opt.n_blocs, opt.n_trials))
for i in xrange(opt.n_blocs):
    opt.model.startBloc()
    for j in xrange(opt.n_trials):
        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
        opt.model.updateValue(opt.responses[i,j])
        delta[i,j] = opt.model.delta


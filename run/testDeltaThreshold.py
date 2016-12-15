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
from fusion_5 import fusion_5
from mixture_5 import mixture_5
from fusion_3 import fusion_3
from fusion_6 import fusion_6
from mixture_6 import mixture_6
from fusion_7 import fusion_7
from mixture_7 import mixture_7
from mixture_3 import mixture_3

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
                    {   '1':fusion_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '2':fusion_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '3':fusion_3(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '4':fusion_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '5':fusion_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '6':fusion_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
                        '7':fusion_7(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True)},
                'mixture': 
                    {   '1':mixture_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '2':mixture_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '3':mixture_3(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '4':mixture_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '5':mixture_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '6':mixture_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '7':mixture_7(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True)},
                'selection':
                    {'1':selection_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1,"eta":0.0001}, 0.05, 10, 0.1, True)}
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
# with open("p_test_last_set.pickle", 'rb') as f:
#     p_test = pickle.load(f)
# with open("timing.pickle", 'rb') as f:
#     timing = pickle.load(f)
with open("p_test_pre_test.pickle", 'rb') as f:
    p_test = pickle.load(f)
with open("timing_pre_test.pickle", 'rb') as f:
    timing = pickle.load(f)

data = {}

for s in p_test.iterkeys():
    m = p_test[s].keys()[0] 
    model = models[m][s[-1]]
    model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test[s][m], sferes=True)        
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
    data[s] = delta

fig = figure(figsize = (10,12))

n = [s for s in p_test.iterkeys() if int(s[-1]) >= 3]
n = np.array(n)[np.argsort([i[-1] for i in n])]

c=1
for s in n:
    m = p_test[s].keys()[0] 
    ax1 = fig.add_subplot(5, 2, c)    
    x = np.arange(len(data[s][0]))
    ax1.set_xticks([])
    ax1.set_title(s[0:-1]+" "+s[-1])
    # plot(data[s][0], '-', color = 'black', label = s[0:-1]+" "+s[-1])
    if int(s[-1]) in [3,4]:
        axhline(p_test[s][m]['shift'])
        in_ = data[s][0]<p_test[s][m]['shift']
        out = data[s][0]>p_test[s][m]['shift']
        plot(x[in_], data[s][0][in_], 'o', color = 'green')
        plot(x[out], data[s][0][out], 'o', color = 'red')
        y1 = data[s][0]
        y2 = np.ones(len(x))*p_test[s][m]['shift']
        # fill_between(x, y1, y2, where=y2 >= y1, facecolor = 'green', interpolate = True)
        # fill_between(x, y1, y2, where=y2 < y1, facecolor = 'red', interpolate = True)
    elif int(s[-1]) in [6]:
        axhline(p_test[s][m]['shift'])
        axhline(-p_test[s][m]['shift'])
        in_ = np.abs(data[s][0])>p_test[s][m]['shift']
        out = np.abs(data[s][0])<p_test[s][m]['shift']
        plot(x[in_], data[s][0][in_], 'o', color = 'green')
        plot(x[out], data[s][0][out], 'o', color = 'red')        
        y1 = data[s][0]
        y2 = np.ones(len(x))*p_test[s][m]['shift']
        y3 = -p_test[s][m]['shift']*np.ones(len(x))
        # fill_between(x, y1, y2, where=y2  y1, facecolor = 'red', interpolate = True)
        # fill_between(x, y1, y2, where=y2 < y1, facecolor = 'green', interpolate = True)        
        # fill_between(x, y1, y3, where=y3 > y1, facecolor = 'green', interpolate = True)        
    if int(s[-1]) in [5, 7]:
        axhline(p_test[s][m]['xi'])
        axhline(p_test[s][m]['shift'])
        in_ = 1.0*(data[s][0]<p_test[s][m]['shift'])+1.0*(data[s][0]>p_test[s][m]['xi'])
        out = 1.0 - in_
        plot(x[in_==1.0], data[s][0][in_==1.0], 'o', color = 'green')
        plot(x[out==1.0], data[s][0][out==1.0], 'o', color = 'red')        
        y1 = data[s][0]
        y2 = p_test[s][m]['shift']*np.ones(len(x))
        y3 = p_test[s][m]['xi']*np.ones(len(x))
        # fill_between(x, y1, y2, where=y2  y1, facecolor = 'red', interpolate = True)
        # fill_between(x, y1, y2, where=y1 < y2 || y1 > y3, facecolor = 'green', interpolate = True)        
        # fill_between(x, y1, y3, where=y3 < y1, facecolor = 'green', interpolate = True)      


    legend()
    ylim(-1.1, 1.1)

    c+=1


plt.savefig("plot3.pdf")


os.system("pdftk soutenance/figure_set_meg_pareto.pdf plot1.pdf plot2.pdf plot3.pdf cat output plot_all_set_meg.pdf")
os.system("rm plot3.pdf")
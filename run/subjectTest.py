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
#import matplotlib.pyplot as plt

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
                        '6':fusion_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True)},
                'mixture': 
                    {   '1':mixture_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '2':mixture_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '4':mixture_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '5':mixture_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
                        '6':mixture_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True)},
                'selection':
                    {'1':selection_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1,"eta":0.0001}, 0.05, 10, 0.1, True)}
            })

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 48
nb_repeat = 200
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


super_data = dict()
super_rt = dict({'model':[]})
hrt = []
hrtm = []
pcrm = dict({'s':[], 'a':[], 'r':[], 't':[], 't2':[]})
allupdate = dict()

for s in p_test.iterkeys():
    # human data
    with open("meg/"+s[0:-1]+".pickle", 'rb') as f:
        data = pickle.load(f)
    m = p_test[s].keys()[0] 
    print s, m, s[-1]       
    model = models[m][s[-1]]
    model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test[s][m])        
    model.startExp()
    for k in xrange(nb_repeat):
        for i in xrange(nb_blocs):
            cats.reinitialize()
            cats.stimuli = np.array(map(_convertStimulus, human.subject[case][s[0:-1]][i+1]['sar'][:,0]))[0:nb_trials]
            model.startBloc()
            for j in xrange(nb_trials):                    
                state = cats.getStimulus(j)
                action = model.chooseAction(state)
                reward = cats.getOutcome(state, action, case=case)                
                model.updateValue(reward)
                # sys.stdin.readline()

    # MODEL
    rtm = np.array(model.reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
    state = convertStimulus(np.array(model.state)).reshape(nb_repeat, nb_blocs, nb_trials)
    action = np.array(model.action).reshape(nb_repeat, nb_blocs, nb_trials)
    responses = np.array(model.responses).reshape(nb_repeat, nb_blocs, nb_trials)
    update = np.array(model.update).reshape(nb_repeat * nb_blocs, nb_trials)
    tmp = np.zeros((nb_repeat, 15))
    for i in xrange(nb_repeat):
        rtm[i] = center(rtm[i], s[0:-1], m) #TODO        
        step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i], case)
        tmp[i] = computeMeanRepresentativeSteps(step)[0]

    pcrm['s'].append(state.reshape(nb_repeat*nb_blocs, nb_trials))
    pcrm['a'].append(action.reshape(nb_repeat*nb_blocs, nb_trials))
    pcrm['r'].append(responses.reshape(nb_repeat*nb_blocs, nb_trials))
    pcrm['t'].append(tmp)        
    pcrm['t2'].append(rtm)
    hrtm.append(np.mean(tmp,0))
    hrt.append(data['mean'][0])
    super_rt['model'].append(m)

    allupdate[s] = extractStimulusPresentation(update, pcrm['s'][-1], pcrm['a'][-1], pcrm['r'][-1])

    
pcr_human = extractStimulusPresentation(human.responses[case], human.stimulus[case], human.action[case], human.responses[case])

for i in pcrm.iterkeys():
    pcrm[i] = np.array(pcrm[i])                
pcrm['s'] = pcrm['s'].reshape(len(p_test.keys())*nb_repeat*nb_blocs, nb_trials)
pcrm['a'] = pcrm['a'].reshape(len(p_test.keys())*nb_repeat*nb_blocs, nb_trials)
pcrm['r'] = pcrm['r'].reshape(len(p_test.keys())*nb_repeat*nb_blocs, nb_trials)
pcrm['t'] = pcrm['t'].reshape(len(p_test.keys())*nb_repeat, 15)

pcr_model = extractStimulusPresentation(pcrm['r'], pcrm['s'], pcrm['a'], pcrm['r'])
rt = (np.mean(pcrm['t'],0), sem(pcrm['t'],0))

ht = np.reshape(human.reaction[case], (len(human.subject[case]), 4*nb_trials))
for i in xrange(len(ht)):
    ht[i] = ht[i]-np.median(ht[i])
    ht[i] = ht[i]/(np.percentile(ht[i], 75)-np.percentile(ht[i], 25))
ht = ht.reshape(len(human.subject[case])*4, nb_trials)    
step, indice = getRepresentativeSteps(ht, human.stimulus[case], human.action[case], human.responses[case], case)
rt_fmri = computeMeanRepresentativeSteps(step) 

#SAVING DATA
data2 = dict()
data2['pcr'] = dict({'model':pcr_model,case:pcr_human})
data2['rt'] = dict({'model':rt,case:rt_fmri})
data2['s'] = dict()
for i, s in zip(xrange(len(p_test.keys())), p_test.keys()):
    data2['s'][s] = dict()
    data2['s'][s]['m'] = hrtm[i]
    data2['s'][s]['h'] = hrt[i]

###############################################
# TO CHANGE HERE 
###############################################

# with open("beh_model_v1.pickle", 'wb') as handle:
#     pickle.dump(data2, handle)

with open("beh_model.pickle", 'wb') as handle:
    pickle.dump(data2, handle)    



fig = figure(figsize = (12, 5))
colors = ['blue', 'red', 'green']
ax1 = fig.add_subplot(1,2,1)
for i in xrange(3):
    plot(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], pcr_model['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    

ax1 = fig.add_subplot(1,2,2)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, color = 'grey', alpha = 0.5)
ax1.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2, color = 'black', alpha = 0.9)

plt.savefig("plot1.pdf")


fig = figure(figsize = (12,20))
for i, s in zip(xrange(11), p_test.keys()):
  ax1 = fig.add_subplot(4,3,i+1)
  ax1.plot(hrt[i], 'o-')
  #ax2 = ax1.twinx()
  ax1.plot(hrtm[i], 'o--', color = 'green')
  ax1.set_title(s+" v"+s[-1]+" "+p_test[s].keys()[0])

plt.savefig("plot2.pdf")

fig = figure(figsize = (12,12))
c = 0
for s in allupdate.keys():    
    if s[-1] == '4' or s[-1] == '3':
        c += 1
        ax1 = fig.add_subplot(3,3,c)
        for i in xrange(3):
            x = range(len(allupdate[s]['mean'][i]))
            y = allupdate[s]['mean'][i]
            e = allupdate[s]['sem'][i]
            errorbar(x, y, e, linewidth = 2, linestyle = '-', color = colors[i])
        ax1.set_title(s[0:-1]+" v4 "+p_test[s].keys()[0])
        ax1.set_ylabel("p(update WM)")

plt.savefig("plot3.pdf")
os.system("pdftk figure_set_meg_pareto.pdf plot1.pdf plot2.pdf plot3.pdf cat output plot_all_set_meg.pdf")
os.system("rm plot1.pdf")
os.system("rm plot2.pdf")
os.system("rm plot3.pdf")

# super_data[o] = data2
# hrt = np.array(hrt)
# hrtm = np.array(hrtm)
# super_rt[o]['rt'] = np.zeros((hrtm.shape[0], hrtm.shape[1], 2))
# super_rt[o]['rt'][:,:,0] = hrt
# super_rt[o]['rt'][:,:,1] = hrtm




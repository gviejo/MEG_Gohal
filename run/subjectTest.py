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

sys.path.append("set_meg_models")
from fusion_2 import fusion_2

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

def center(x):
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x
# def center(x, o, s, m):    
#     x = x-timing[o][s][m][0]
#     x = x/timing[o][s][m][1]
#     return x

# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 48
nb_repeat = 1
case = 'meg'

cats = CATS(nb_trials)

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("p_test_last_set.pickle", 'r') as f:
  p_test = pickle.load(f)


model = fusion_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])

m = p_test['S74'].keys()[0]    
model.setAllParameters(p_test['S74'][m])
model.startExp()
for k in xrange(nb_repeat):
    for i in xrange(nb_blocs):
        cats.reinitialize()
        #cats.stimuli = np.array(map(_convertStimulus, human.subject[case][s][i+1]['sar'][:,0]))[0:nb_trials]
        model.startBloc()
        for j in xrange(nb_trials):                    
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action, case=case)
            model.updateValue(reward)

# MODEL
rtm = np.array(model.reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
state = convertStimulus(np.array(model.state)).reshape(nb_repeat, nb_blocs, nb_trials)
action = np.array(model.action).reshape(nb_repeat, nb_blocs, nb_trials)
responses = np.array(model.responses).reshape(nb_repeat, nb_blocs, nb_trials)
tmp = np.zeros((nb_repeat, 15))
for i in xrange(nb_repeat):
    # rtm[i] = center(rtm[i], o, s, m)
    rtm[i] = center(rtm[i])
    step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i], case)
    tmp[i] = computeMeanRepresentativeSteps(step)[0]









sys.exit()

super_data = dict()
super_rt = dict()


hrt = []
hrtm = []
pcrm = dict({'s':[], 'a':[], 'r':[], 't':[], 't2':[]})

for s in p_test.iterkeys():        
    m = p_test[o][s].keys()[0]    
    models[m].setAllParameters(p_test[o][s][m])
    models[m].startExp()
    for k in xrange(nb_repeat):
        for i in xrange(nb_blocs):
            cats.reinitialize()
            #cats.stimuli = np.array(map(_convertStimulus, human.subject[case][s][i+1]['sar'][:,0]))[0:nb_trials]
            models[m].startBloc()
            for j in xrange(nb_trials):                    
                state = cats.getStimulus(j)
                action = models[m].chooseAction(state)
                reward = cats.getOutcome(state, action, case=case)
                models[m].updateValue(reward)

    # MODEL
    rtm = np.array(models[m].reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
    state = convertStimulus(np.array(models[m].state)).reshape(nb_repeat, nb_blocs, nb_trials)
    action = np.array(models[m].action).reshape(nb_repeat, nb_blocs, nb_trials)
    responses = np.array(models[m].responses).reshape(nb_repeat, nb_blocs, nb_trials)
    tmp = np.zeros((nb_repeat, 15))
    for i in xrange(nb_repeat):
        rtm[i] = center(rtm[i], o, s, m)
        step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i], case)
        tmp[i] = computeMeanRepresentativeSteps(step)[0]

    pcrm['s'].append(state.reshape(nb_repeat*nb_blocs, nb_trials))
    pcrm['a'].append(action.reshape(nb_repeat*nb_blocs, nb_trials))
    pcrm['r'].append(responses.reshape(nb_repeat*nb_blocs, nb_trials))
    
    pcrm['t'].append(tmp)        
    pcrm['t2'].append(rtm)
    hrtm.append(np.mean(tmp,0))
    hrt.append(data['mean'][0])
        

    super_rt[o]['model'].append(m)

    
    pcr_human = extractStimulusPresentation(human.responses[case], human.stimulus[case], human.action[case], human.responses[case])

    for i in pcrm.iterkeys():
        pcrm[i] = np.array(pcrm[i])                
    pcrm['s'] = pcrm['s'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['a'] = pcrm['a'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['r'] = pcrm['r'].reshape(len(p_test[o].keys())*nb_repeat*nb_blocs, nb_trials)
    pcrm['t'] = pcrm['t'].reshape(len(p_test[o].keys())*nb_repeat, 15)
    
    pcr_model = extractStimulusPresentation(pcrm['r'], pcrm['s'], pcrm['a'], pcrm['r'])
    rt = (np.mean(pcrm['t'],0), sem(pcrm['t'],0))

    # ht = np.reshape(human.reaction[case], (14, 4*39))
    # for i in xrange(len(ht)):
    #     ht[i] = ht[i]-np.median(ht[i])
    #     ht[i] = ht[i]/(np.percentile(ht[i], 75)-np.percentile(ht[i], 25))
    # ht = ht.reshape(14*4, 39)    
    # step, indice = getRepresentativeSteps(ht, human.stimulus[case], human.action[case], human.responses[case])
    # rt_fmri = computeMeanRepresentativeSteps(step) 

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
    for i, s in zip(xrange(len(p_test[o].keys())), p_test[o].keys()):
        data2['s'][s] = dict()
        data2['s'][s]['m'] = hrtm[i]
        data2['s'][s]['h'] = hrt[i]



    fig = figure(figsize = (15, 12))
    colors = ['blue', 'red', 'green']
    ax1 = fig.add_subplot(4,4,1)
    for i in xrange(3):
        plot(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
        errorbar(range(1, len(pcr_model['mean'][i])+1), pcr_model['mean'][i], pcr_model['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
        plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    


    ax1 = fig.add_subplot(4,4,2)
    ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, color = 'grey', alpha = 0.5)
    ax1.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2, color = 'black', alpha = 0.9)


    for i, s in zip(xrange(14), p_test[o].keys()):
      ax1 = fig.add_subplot(4,4,i+3)
      ax1.plot(hrt[i], 'o-')
      #ax2 = ax1.twinx()
      ax1.plot(hrtm[i], 'o--', color = 'green')
      ax1.set_title(s+" "+p_test[o][s].keys()[0])

    show()

    super_data[o] = data2
    hrt = np.array(hrt)
    hrtm = np.array(hrtm)
    super_rt[o]['rt'] = np.zeros((hrtm.shape[0], hrtm.shape[1], 2))
    super_rt[o]['rt'][:,:,0] = hrt
    super_rt[o]['rt'][:,:,1] = hrtm

saving = raw_input("Save ? : y/n : ")
if saving == "y":
    with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_model.pickle") , 'wb') as handle:    
         pickle.dump(super_data, handle)
    with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rts_all_subjects.pickle") , 'wb') as handle:    
         pickle.dump(super_rt, handle)
    with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/choice_rt_all_data.pickle"), 'wb') as handle:
	    pickle.dump(pcrm, handle)


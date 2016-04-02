#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test choice only extremum of pareto fronts

run subjectTestExtremum.py

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys,os

from optparse import OptionParser
import numpy as np

sys.path.append("../src")
from fonctions import *
from ColorAssociationTasks import CATS
# from HumanLearning import HLearning
from Models import *
from Selection import *
from matplotlib import *
from pylab import *
import pickle
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import leastsq
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------


# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
		return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING / LOAD MEG FILES
# -----------------------------------
# human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
with open("meg.pickle", 'rb') as f:
	human = pickle.load(f)
# REMOVING S1
human.pop('S1')
keys = human.keys()
skeys = []
for s in keys:
	for i in xrange(4):
		skeys.append(s+str(i+1))
human['reward'] = np.array([human[s]['reward'] for s in keys]).reshape(len(keys)*4,48)
human['state'] = np.array([human[s]['state'] for s in keys]).reshape(len(keys)*4,48)
human['action'] = np.array([human[s]['action'] for s in keys]).reshape(len(keys)*4,48)


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 48
nb_repeat = 20
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
				"qlearning":QLearning(cats.states, cats.actions),
				"bayesian":BayesianWorkingMemory(cats.states, cats.actions),
				"selection":KSelection(cats.states, cats.actions),
				"mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("extremum.pickle", 'r') as f:
	p_test = pickle.load(f)

# REMOVING S1
for g in p_test.keys():
	p_test[g].pop('S1')

colors_m = dict({'fusion':'#F1433F',
				'bayesian': '#D5A253',
				'qlearning': '#6E8243',
				'selection':'#70B7BA',
				'mixture':'#3D4C53'})

# entropy = {'Hb':{},'Hf':{}}

# pcrm = 
pcrm = dict()
pcr = dict()
choice_only_all_data = dict()


states = []
actions = []
responses = []
# p_test = dict({'log':p_test})

for o in p_test.iterkeys():
	pcrm[o] = dict({'s':[], 'a':[], 'r':[]})
	choice_only_all_data[o] = dict()
	for s in p_test[o].iterkeys():
		m = p_test[o][s].keys()[0]
		print "Testing "+s+" with "+m+" selected by "+o
		models[m].setAllParameters(p_test[o][s][m])
		# on ajoute sigma pour eviter de bugger sur les temps de réaction
		if m == 'selection': 
			models[m].parameters.update({'sigma_rt':1.0})
		else :
			models[m].parameters.update({'sigma':1.0})
		models[m].startExp()
		for k in xrange(nb_repeat):
			for i in xrange(nb_blocs):
				cats.reinitialize()
				# cats.stimuli = np.array(map(_convertStimulus, human.subject['meg'][s][i+1]['sar'][:,0]))                    
				models[m].startBloc()
				for j in xrange(nb_trials):
					state = cats.getStimulus(j)
					action = models[m].chooseAction(state)
					reward = cats.getOutcome(state, action, case='meg')                    
					models[m].updateValue(reward)            

		state = convertStimulus(np.array(models[m].state))
		action = np.array(models[m].action)
		responses = np.array(models[m].responses)                    
		pcrm[o]['s'].append(state)
		pcrm[o]['a'].append(action)
		pcrm[o]['r'].append(responses)
		choice_only_all_data[o][s] = dict({'s':state,'a':action,'r':responses})
		# hall = np.array(models[m].Hall)
		# if hall[:,:,0].sum():
		#     entropy['Hb'][s] = {m:extractStimulusPresentation(hall[:,:,0], state, action, responses)}

		# if hall[:,:,1].sum():
		#     entropy['Hf'][s] = {m:extractStimulusPresentation(hall[:,:,1], state, action, responses)}
	for i in pcrm[o].iterkeys():    
		pcrm[o][i] = np.array(pcrm[o][i])
		pcrm[o][i] = np.reshape(pcrm[o][i], (pcrm[o][i].shape[0]*pcrm[o][i].shape[1], pcrm[o][i].shape[2]))
	pcr[o] = extractStimulusPresentation(pcrm[o]['r'], pcrm[o]['s'], pcrm[o]['a'], pcrm[o]['r'])
	choice_only_all_data[o]['pcrm'] = pcrm[o]
	

		
pcr_human = extractStimulusPresentation(human['reward'], human['state'], human['action'], human['reward'])
pcr['meg'] = pcr_human

fig = figure()
colors = ['blue', 'red', 'green']
# ax1 = fig.add_subplot(1,3,1)
for j,o in zip([1,2], ['log', 'bic']):
	ax1 = fig.add_subplot(1,2,j)
	for i in xrange(3):
		plot(range(1, len(pcr[o]['mean'][i])+1), pcr[o]['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
		errorbar(range(1, len(pcr[o]['mean'][i])+1), pcr[o]['mean'][i], pcr[o]['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
		plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
		#errorbar(range(1, len(pcr[o]_human['mean'][i])+1), pcr[o]_human['mean'][i], pcr[o]_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)
	ax1.set_title(o)
	ax1.set_xlabel("Trials")
	ax1.set_ylabel("Performance")

show()

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_choice_only.pickle") , 'wb') as handle:    
#      pickle.dump(pcr, handle)

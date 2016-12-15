#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import os
from optparse import OptionParser
import numpy as np
import scipy.io as sio
sys.path.append("../src")
from fonctions import *
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *
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
# HUMAN LEARNING
# -----------------------------------
# with open("meg.pickle", 'rb') as f:
# 	human = pickle.load(f)
# # REMOVING S1
# human.pop('S1')
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../PEPS_GoHaL/fMRI',39)}))
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
# VARIABLES NAME 
# -----------------------------------
variables = {'bayesian':['p_a', 'Hb', 'N', 'Q'],
			'qlearning':['p_a', 'Hf', 'Q', 'delta'],
			'fusion':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'p_sigmoide', 'p_decision', 'Hb_steps'],
			'selection':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'vpi', 'r_rate'],
			'mixture':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'w', ]}
# -----------------------------------
# -----------------------------------
# LOADING PARAMETERS
# -----------------------------------
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


# -----------------------------------
# GENERATE REgressors
# -----------------------------------
for s in p_test.iterkeys():
	m = p_test[s].keys()[0]
	data = np.zeros(4, dtype = {'names':variables[m], 'formats':['O']*len(variables[m])})
	m = p_test[s].keys()[0] 
	print s, m, s[-1]       
	model = models[m][s[-1]]
	model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test[s][m], sferes = True)            
	model.value = np.zeros((4,60))
	model.reaction = np.zeros((4,60))				
	for i in xrange(4):
		model.startBloc()
		nb_trials = human.subject['meg'][s[0:-1]][i+1]['sar'].shape[0]
		for v in variables[m]:
			data[v][i] = []		
		for j in xrange(nb_trials):
			state, action, reward = human.subject['meg'][s[0:-1]][i+1]['sar'][j]
			model.computeValue(state-1, action-1, (i,j))				
			data[i]['p_a'].append(model.p_a.copy())
			data[i]['Q'].append(model.q_values.copy())
			model.updateValue(reward)
			if 'N' in variables[m]:
				data[i]['N'].append(model.N)
				data[i]['Hb'].append(model.Hb)
			if 'delta' in variables[m]:
				data[i]['delta'].append(model.delta)
				data[i]['Hf'].append(model.Hf)					
			if m == 'selection':
				data[i]['vpi'].append(model.vpi)
				data[i]['r_rate'].append(model.r_rate)
			if m == 'mixture':
				data[i]['w'].append(model.w[model.current_state])
			if m == 'fusion':
				data[i]['p_sigmoide'].append(model.p_sigmoide)
				data[i]['p_decision'].append(model.p_decision)
				data[i]['Hb_steps'].append(model.evolution_entropy)
	for i in xrange(4):
		for v in variables[m]:
			data[i][v] = np.array(data[i][v])		
	# SAVING
	# sio.savemat("../MATLAB/"+s+"/"+m+".mat", {m : data}, oned_as='column')	
	os.system("mkdir ~/Dropbox/PEPS_GoHaL/Beh_Model/regressors_15_12_2016/"+s[0:-1])
	sio.savemat(os.path.expanduser("~/Dropbox/PEPS_GoHaL/Beh_Model/regressors_15_12_2016/"+s[0:-1]+"/"+m+"_"+s[-1]+".mat"), {m : data}, oned_as='column')



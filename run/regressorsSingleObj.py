#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
regressors for choice only


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
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

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args()


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human2 = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',48)}))
with open("meg.pickle", 'rb') as f:
	human = pickle.load(f)
# REMOVING S1
human.pop('S1')
# -----------------------------------
# VARIABLES NAME 
# -----------------------------------
variables = {'bayesian':['p_a', 'Hb', 'Q'],
			'qlearning':['p_a', 'Hf', 'Q', 'delta'],
			'fusion':['p_a', 'Hb', 'Hf','Q', 'delta', 'p_dec', 'p_ret', 'p_sig', 'p_at'],
			'selection':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'vpi', 'r_rate'],
			'mixture':['p_a', 'Hb', 'Hf', 'Q', 'delta', 'w', ]}
# -----------------------------------
# -----------------------------------
# LOADING DATA
# -----------------------------------
front = pareto(options.input, case = 'meg')

# ----------------------------------
# Best parameters for each model and each subject for the choice objectives only
# ----------------------------------
best_parameters = {}
subjects = human.keys()

for s in subjects:
	best_parameters[s] = {}
	for m in front.data.keys():
		best_parameters[s][m] = {}
		data = []
		for i in front.data[m][s].iterkeys():
			tmp = np.hstack((np.ones((len(front.data[m][s][i]),1))*i,front.data[m][s][i]))
			data.append(tmp)
		data = np.vstack(data)
		data[:,3]-=2000.0
		best_ind = np.argmax(data[:,3])		
		best_parameters[s][m] = dict(zip(front.p_order[m][0:],data[best_ind,5:]))                

# ---------------------------------
# Testing
# ---------------------------------
models = front.pareto.keys()
for s in subjects:
	# os.system("mkdir matlab/"+s)
	# for m in front.data.keys():		
	for m in ['fusion']:
		print s, m
		data = np.zeros(4, dtype = {'names':variables[m], 'formats':['O']*len(variables[m])})						
		parameters = best_parameters[s][m]
		front.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)		
		front.models[m].value = np.zeros((4,60))
		front.models[m].reaction = np.zeros((4,60))				
		for i in xrange(4):
			front.models[m].startBloc()
			nb_trials = human2.subject['meg'][s][i+1]['sar'].shape[0]
			for v in variables[m]:
				data[v][i] = []
			for j in xrange(nb_trials):
				state, action, reward = human2.subject['meg'][s][i+1]['sar'][j]
				front.models[m].computeValue(state-1, action-1, (i,j))												
				front.models[m].updateValue(reward)
				
				data[i]['p_a'].append(front.models[m].p_a.copy())
				data[i]['Q'].append(front.models[m].q_values.copy())

				if m == 'fusion':					
					data[i]['p_sig'].append(front.models[m].p_sigmoide.copy())
					data[i]['p_ret'].append(front.models[m].p_retrieval)
					data[i]['p_dec'].append(front.models[m].p_decision)
					data[i]['p_at'].append(front.models[m].p_actions)
				if 'Hb' in variables[m]:					
					data[i]['Hb'].append(front.models[m].evolution_entropy)
				if 'delta' in variables[m]:
					data[i]['delta'].append(front.models[m].delta)
					data[i]['Hf'].append(front.models[m].Hf)					
				if m == 'selection':
					data[i]['vpi'].append(front.models[m].vpi)
					data[i]['r_rate'].append(front.models[m].r_rate)
				if m == 'mixture':
					data[i]['w'].append(front.models[m].w[front.models[m].current_state])		
		for i in xrange(4):
			for v in variables[m]:
				data[i][v] = np.array(data[i][v])		
		# SAVING		
		# sio.savemat("matlab/"+s+"/"+m+".mat", {m : data}, oned_as='column')	
		sio.savemat(os.path.expanduser("~/Dropbox/PEPS_GoHaL/Beh_Model/"+s+"/"+m+".mat"), {m : data}, oned_as='column')


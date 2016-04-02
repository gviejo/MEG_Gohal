#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
import scipy.io as sio
sys.path.append("../../src")
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
def rankTchebytchebv(tmp, lambdaa = 0.5, epsilon = 0.001):
	ideal = np.max(tmp, 0)
	nadir = np.min(tmp, 0)
	value = lambdaa*((ideal-tmp)/(ideal-nadir))
	value = np.max(value,1)+epsilon*np.sum(value,1)
	return value

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
with open("meg.pickle", 'rb') as f:
	human = pickle.load(f)
# REMOVING S1
human.pop('S1')
# -----------------------------------
# VARIABLES NAME 
# -----------------------------------
variables = {'bayesian':['p_a', 'Hb', 'N', 'Q'],
			'qlearning':['p_a', 'Hf', 'Q', 'delta'],
			'fusion':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta'],
			'selection':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'vpi', 'r_rate'],
			'mixture':['p_a', 'Hb', 'Hf', 'N', 'Q', 'delta', 'w', ]}
# -----------------------------------
# -----------------------------------
# LOADING DATA
# -----------------------------------
front = pareto(options.input, case = 'meg')
front.constructParetoFrontier('r2') # 'r2', 'bic', 'aic' , 'log'
front.removeIndivDoublons()
front.constructMixedParetoFrontier()
front.rankTchebytchev()
front.timeConversion()

with open("parameters_meg.pickle", 'wb') as handle:
	pickle.dump(front.p_test, handle)

with open("timing_meg.pickle", 'wb') as f:
	pickle.dump(front.timing, f)

sys.exit()
models = front.pareto.keys()
for s in human.subject['meg'].keys():
	for m in models:	
		print s, m
		data = np.zeros(4, dtype = {'names':variables[m], 'formats':['O']*len(variables[m])})				
		value = rankTchebytchebv(front.pareto[m][s][:,3:5])
		parameters = dict({front.p_order[m][i]:front.pareto[m][s][np.argmin(value),5:][i] for i in xrange(len(front.p_order[m]))})
		front.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)		
		front.models[m].value = np.zeros((4,60))
		front.models[m].reaction = np.zeros((4,60))				
		for i in xrange(4):
			front.models[m].startBloc()
			nb_trials = human.subject['meg'][s][i+1]['sar'].shape[0]
			for v in variables[m]:
				data[v][i] = []
			for j in xrange(nb_trials):
				state, action, reward = human.subject['meg'][s][i+1]['sar'][j]
				front.models[m].computeValue(state-1, action-1, (i,j))				
				data[i]['p_a'].append(front.models[m].p_a.copy())
				data[i]['Q'].append(front.models[m].q_values.copy())
				front.models[m].updateValue(reward)
				if 'N' in variables[m]:
					data[i]['N'].append(front.models[m].N)
					data[i]['Hb'].append(front.models[m].Hb)
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
		# sio.savemat("../MATLAB/"+s+"/"+m+".mat", {m : data}, oned_as='column')	
		sio.savemat(os.path.expanduser("~/Dropbox/PEPS_GoHaL/Beh_Model/"+s+"/"+m+".mat"), {m : data}, oned_as='column')



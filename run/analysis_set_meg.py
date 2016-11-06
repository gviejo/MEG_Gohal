#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and write for last_set for manuscrit


"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../src")

from Models import *
from Selection import *

#from matplotlib import *
#from pylab import *

from Sferes import pareto
from itertools import *
from time import sleep
import cPickle as pickle

p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift'], 
					'qlearning':['alpha','beta', 'sigma', 'kappa', 'shift'],
					'bayesian':['length','noise','threshold', 'sigma'],
					'selection':['beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
					'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift']
					}) 

front = pareto("", 'meg') # dummy for rt

models = dict({ "fusion" 	:	FSelection				(front.states, front.actions),
                "qlearning"	:	QLearning				(front.states, front.actions),
                "bayesian"	:	BayesianWorkingMemory	(front.states, front.actions),
                "selection"	:	KSelection				(front.states, front.actions),
                "mixture"	:	CSelection				(front.states, front.actions)})


# -----------------------------------
# LOADING DATA
# -----------------------------------
sujet = front.human.keys()
sujet.remove("S1")
id_to_models = dict({	1:'fusion',
						2:'mixture',
						3:'bayesian',
						4:'qlearning',
						5:'selection'})
models_to_id = dict({	'fusion':		1,
						'mixture':		2,
						'bayesian':		3,
						'qlearning':	4,
						'selection':	5})

set_to_models = dict({	1:[1,2,3,4,5],
						2:[1,2,4],
						3:[1,2],
						4:[1,2]})

n_run = 3
data = {}
pareto = {}
pareto2 = {}
pareto3 = {}
p_test = {}
p_test2 = {}
tche = {}
indd = {}
position = {}

#------------------------------------
# best log/rt
#------------------------------------
best_log = -4*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2))
worst_log = front.N*np.log(0.2)


# ------------------------------------
# LOAD DATA
# ------------------------------------
for s in sujet: 
	data[s] = dict()
	pareto[s] = dict() # first pareto set
	pareto2[s] = dict() # second pareto set with the set dimension
	pareto3[s] = dict() # third pareto set with mixed models
	# for p in set_to_models.iterkeys(): # ensemble testé
	for p in [1,2,3,4]: # ensemble testé
		data[s][p] = dict()
		pareto[s][p] = dict()		
		for m in set_to_models[p]: # modele dans ensemble testé
			data[s][p][id_to_models[m]] = dict()
			pareto[s][p][id_to_models[m]] = dict()
			for r in xrange(n_run):						
				data[s][p][id_to_models[m]][r] = np.genfromtxt("set_meg/set_"+str(p)+"_"+str(m)+"/sferes_"+id_to_models[m]+"_meg_inserm_"+s+"_"+str(r)+"_"+str(p)+".dat")
				order = p_order[id_to_models[m]]
				scale = models[id_to_models[m]].bounds
				for i in order:
					data[s][p][id_to_models[m]][r][:,order.index(i)+4] = scale[i][0]+data[s][p][id_to_models[m]][r][:,order.index(i)+4]*(scale[i][1]-scale[i][0])

			part = data[s][p][id_to_models[m]]
			tmp={n:part[n][part[n][:,0]==np.max(part[n][:,0])] for n in part.iterkeys()}			
			tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])			
			ind = tmp[:,3] != 0
			tmp = tmp[ind]
			tmp = tmp[tmp[:,3].argsort()][::-1]
			pareto_frontier = [tmp[0]]
			for pair in tmp[1:]:
				if pair[4] >= pareto_frontier[-1][4]:
					pareto_frontier.append(pair)
			pareto[s][p][id_to_models[m]] = np.array(pareto_frontier)
			pareto[s][p][id_to_models[m]][:,3] = pareto[s][p][id_to_models[m]][:,3] - 2000.0
			pareto[s][p][id_to_models[m]][:,4] = pareto[s][p][id_to_models[m]][:,4] - 500.0            
			# bic
			# pareto[s][p][id_to_models[m]][:,3] = 2*pareto[s][p][id_to_models[m]][:,3] - float(len(p_order[id_to_models[m]]))*np.log(front.N)
			# best_bic = 2*best_log - float(len(p_order[id_to_models[m]]))*np.log(front.N)			
			# worst_bic = 2*worst_log - float(len(p_order[id_to_models[m]]))*np.log(front.N)                    
			# pareto[s][p][id_to_models[m]][:,3] = (pareto[s][p][id_to_models[m]][:,3]-worst_bic)/(best_bic-worst_bic)			
			# r2
			pareto[s][p][id_to_models[m]][:,3] = 1.0 - (pareto[s][p][id_to_models[m]][:,3]/(front.N*np.log(0.2)))
			# rt
			# if s == 'p':
			# 	pareto[s][p][id_to_models[m]][:,4] = 1.0 - ((-pareto[s][p][id_to_models[m]][:,4])/(8.0*np.power(2.0*front.rt_reg_monkeys[s][:,1], 2).sum()))
			# else :
			pareto[s][p][id_to_models[m]][:,4] = 1.0 - ((-pareto[s][p][id_to_models[m]][:,4])/(2.0*np.power(2.0*front.human[s]['mean'][0], 2).sum()))
	
# --------------------------------------
# MIXED PARETO FRONTIER between sets
# --------------------------------------
	for m in id_to_models.iterkeys():
		tmp = {}	
		# for p in set_to_models.iterkeys():
		for p in [1,2,3,4]:
			if m in set_to_models[p]:
				tmp[p] = pareto[s][p][id_to_models[m]]
		tmp=np.vstack([np.hstack((np.ones((len(tmp[p]),1))*p,tmp[p])) for p in tmp.iterkeys()])			
		ind = tmp[:,4] != 0
		tmp = tmp[ind]
		tmp = tmp[tmp[:,4].argsort()][::-1]
		pareto_frontier = [tmp[0]]		
		for pair in tmp[1:]:
			if pair[5] >= pareto_frontier[-1][5]:
				pareto_frontier.append(pair)		
		pareto2[s][id_to_models[m]] = np.array(pareto_frontier)


# -------------------------------------
# MIXED PARETO FRONTIER between models
# ------------------------------------
	tmp = []
	for m in pareto2[s].iterkeys():		
		tmp.append(np.hstack((np.ones((len(pareto2[s][m]),1))*models_to_id[m], pareto2[s][m][:,0:6])))            	
	tmp = np.vstack(tmp)
	tmp = tmp[tmp[:,5].argsort()][::-1]                        
	if len(tmp):
		pareto3[s] = []
		pareto3[s] = [tmp[0]]
		for pair in tmp[1:]:
			if pair[6] >= pareto3[s][-1][6]:
				pareto3[s].append(pair)
		pareto3[s] = np.array(pareto3[s])            	
# pareto2 =   set | run | gen | num | fit1 | fit2				
# pareto3 = model | set | run | gen | num | fit1 | fit2		
# -------------------------------------
# TCHEBYTCHEV
# -------------------------------------	
	tmp = pareto3[s][:,5:]
	tmp = tmp[(tmp[:,0]>0)*(tmp[:,1]>0)]
	ideal = np.max(tmp, 0)
	nadir = np.min(tmp, 0)
	value = 0.5*((ideal-tmp)/(ideal-nadir))
	value = np.max(value, 1)+0.001*np.sum(value,1)
	tche[s] = value
	ind_best_point = np.argmin(value)
	# Saving best individual
	best_ind = pareto3[s][ind_best_point]
	indd[s] = best_ind	
	
	
	# from data dictionnary
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])

	print s
	print "set ", set_
	print "run ", run_
	print "gen ", gen_
	print "num ", num_
	print "model" , m

	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test[s+str(set_)] = dict({m:dict(zip(p_order[m],tmp[4:]))})                        
	position[s+str(set_)] = best_ind[5:]

# ------------------------------------
# BEST RT
# ------------------------------------
	index = (pareto3[s][:,5] > 0)*(pareto3[s][:,6] > 0)
	tmp = pareto3[s][index,:]
	best_ind = tmp[-1]
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])

	# print s
	# print "set ", set_
	# print "run ", run_
	# print "gen ", gen_
	# print "num ", num_
	# print "model" , m

	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test2[s+str(set_)] = dict({m:dict(zip(p_order[m],tmp[4:]))})
	



# SAVING IN DROPBOX
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/pareto2.pickle", 'wb') as f:
# 	pickle.dump(pareto2, f)
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/pareto3.pickle", 'wb') as f:
# 	pickle.dump(pareto3, f)
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/position.pickle", 'wb') as f:
# 	pickle.dump(position, f)

with open("p_test_last_set.pickle", 'wb') as f:
	pickle.dump(p_test, f)
with open("p_test2_last_set.pickle", 'wb') as f:
	pickle.dump(p_test2, f)


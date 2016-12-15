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
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from Selection import *

#from matplotlib import *
#from pylab import *

from Sferes import pareto, EA
from itertools import *
from time import sleep
import cPickle as pickle

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

# ------------------------------------


p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift', 'xi'], 
					'qlearning':['alpha','beta', 'sigma', 'kappa', 'shift'],
					'bayesian':['length','noise','threshold', 'sigma'],
					'selection':['beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
					'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift', 'xi']
					}) 
p_order_5 = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift', 'xi'], 
					'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift', 'xi']
					}) 


front = pareto("", 'meg') # dummy for rt

models = dict({ "fusion" 	:	FSelection				(front.states, front.actions),
				"qlearning"	:	QLearning				(front.states, front.actions),
				"bayesian"	:	BayesianWorkingMemory	(front.states, front.actions),
				"selection"	:	KSelection				(front.states, front.actions),
				"mixture"	:	CSelection				(front.states, front.actions)})

models_set = dict({'fusion':
					{	1:fusion_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						2:fusion_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						3:fusion_3(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						4:fusion_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						5:fusion_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						6:fusion_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True),
						7:fusion_7(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1}, True)},
				'mixture': 
					{	1:mixture_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						2:mixture_2(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						3:mixture_3(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						4:mixture_4(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						5:mixture_5(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						6:mixture_6(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True),
						7:mixture_7(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {'length':1, 'weight':0.5}, True)},
				'selection':
					{1:selection_1(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], {"length":1,"eta":0.0001}, 0.05, 10, 0.1, True)}
			})
nb_parameters = dict({'fusion':{		1:8,
											2:9,
											3:9,
											4:10,
											5:11,
											6:10,
											7:9	},
							'mixture':{		1:7,
											2:8,
											3:8,
											4:9,
											5:10,
											6:9,
											7:8		},
							'bayesian':{	1:4		},
							'qlearning':{	1:3,
											2:4		},
							'selection':{	1:7		}
							})



# -----------------------------------
# LOADING DATA
# -----------------------------------
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../PEPS_GoHaL/fMRI',39)}))
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
						4:[1,2],
						5:[1,2],
						6:[1,2],
						7:[1,2]})

n_run = 3
data = {}
pareto = {}
pareto2 = {}
pareto3 = {}
pareto4 = {}
pareto5 = {}
p_test = {}
p_test2 = {}
p_test_v1 = {}
p_test_pre_test = {}
tche = {}
indd = {}
position = {}
position_pre_test = {}
timing = {}
timing_v1 = {}
timing_all = {}
timing_pre_test = {}
#------------------------------------
# best log/rt
#------------------------------------
best_log = -4*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2))
worst_log = front.N*np.log(0.2)


# ------------------------------------
# LOAD DATA
# ------------------------------------
for s in sujet: 
# for s in ['S3']:
	data[s] = dict()
	pareto[s] = dict() # first pareto set
	pareto2[s] = dict() # second pareto set with the set dimension
	pareto3[s] = dict() # third pareto set with mixed models
	# for p in set_to_models.iterkeys(): # ensemble testé
	for p in [1,2,3,4,5,6,7]: # ensemble testé
		data[s][p] = dict()
		pareto[s][p] = dict()		
		for m in set_to_models[p]: # modele dans ensemble testé
			data[s][p][id_to_models[m]] = dict()
			pareto[s][p][id_to_models[m]] = dict()
			for r in xrange(n_run):						
				data[s][p][id_to_models[m]][r] = np.genfromtxt("set_meg_all_rt_points/set_"+str(p)+"_"+str(m)+"/sferes_"+id_to_models[m]+"_meg_inserm_"+s+"_"+str(r)+"_"+str(p)+".dat")
				if m in [3,4,5]:
				#if p in [1,2,3,4]:				
					order = p_order[id_to_models[m]]
					scale = models[id_to_models[m]].bounds
					for i in order:
						data[s][p][id_to_models[m]][r][:,order.index(i)+4] = scale[i][0]+data[s][p][id_to_models[m]][r][:,order.index(i)+4]*(scale[i][1]-scale[i][0])
				elif m in [1,2]:
					order = p_order[id_to_models[m]]
					if p in [1,2,3,4]:
						scale = models[id_to_models[m]].bounds
						scale['xi'] = [0,1]
					elif p in [5,6,7]:					
						scale = models_set[id_to_models[m]][p].bounds
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
			pareto[s][p][id_to_models[m]][:,3] = 2*pareto[s][p][id_to_models[m]][:,3] - nb_parameters[id_to_models[m]][p]*np.log(front.N)
			best_bic = 2*best_log - nb_parameters[id_to_models[m]][p]*np.log(front.N)
			worst_bic = 2*worst_log - nb_parameters[id_to_models[m]][p]*np.log(front.N)
			pareto[s][p][id_to_models[m]][:,3] = (pareto[s][p][id_to_models[m]][:,3]-worst_bic)/(best_bic-worst_bic)			
			# r2
			# pareto[s][p][id_to_models[m]][:,3] = 1.0 - (pareto[s][p][id_to_models[m]][:,3]/(front.N*np.log(0.2)))
			# rt
			pareto[s][p][id_to_models[m]][:,4] = 1.0 - ((-pareto[s][p][id_to_models[m]][:,4])/(2.0*np.power(2.0*front.human[s]['mean'][0], 2).sum()))			

# --------------------------------------
# MIXED PARETO FRONTIER between sets
# --------------------------------------
	for m in id_to_models.iterkeys():
		tmp = {}	
		# for p in set_to_models.iterkeys():
		for p in [1,2,3,4,5,6,7]:
			if m in set_to_models[p]:				
				tmp[p] = pareto[s][p][id_to_models[m]]
				#if p in [1,2,3,4]: # VERY BAD
				#	tmp[p] = np.hstack((tmp[p],np.ones((len(tmp[p]),1))))
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
	positif = (tmp[:,0]>0)*(tmp[:,1]>0)
	tmp = tmp[positif]
	# ideal = np.max(tmp, 0)
	# nadir = np.min(tmp, 0)
	# value = 0.5*((ideal-tmp)/(ideal-nadir))
	# value = np.max(value, 1)+0.001*np.sum(value,1)
	# tche[s] = value
	# ind_best_point = np.argmin(value)
	# # Saving best individual
	# best_ind = pareto3[s][positif][ind_best_point]
	# indd[s] = best_ind	

# -------------------------------------
# Doing OWA INSTEAD
# -------------------------------------
	value = np.sum(np.sort(tmp)*[0.9, 0.1], 1)
	tche[s] = value
	ind_best_point = np.argmax(value)
	# Saving best individual
	best_ind = pareto3[s][positif][ind_best_point]
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
	#if set_ in [1,2,3,4]:				
	#	order = p_order[m]
	#	p_test[s+str(set_)] = dict({m:dict(zip(order,tmp[4:]))})
		# PROBLEM HERE                        
	#elif set_ in [5,6]:
	order = p_order_5[m]
	p_test[s+str(set_)] = dict({m:dict(zip(order,tmp[4:]))})                        

	position[s+str(set_)] = best_ind[5:]
# -----------------------------------
# CHECKING PYTHON MODELS + SAVING RT TIMING
# -----------------------------------	
	print "from sferes :", tmp[2]-2000, tmp[3]-500
	timing [s] = dict({})
	model = models_set[m][set_]
	model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test[s+str(set_)][m], sferes = True)        
	with open("meg/"+s+".pickle", "rb") as f:
		data2 = pickle.load(f)
	opt = EA(data2, s, model)                                
	for i in xrange(opt.n_blocs):
		opt.model.startBloc()
		for j in xrange(opt.n_trials):
			opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
			opt.model.updateValue(opt.responses[i,j])
	opt.fit[0] = float(np.sum(opt.model.value))
	timing[s][m] = [np.median(opt.model.reaction)]
	opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
	timing[s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
	opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
	timing[s][m] = np.array(timing[s][m])
	opt.fit[1] = float(-opt.leastSquares())
	
	print "from test   :", opt.fit[0], opt.fit[1], "\n"
	
# -------------------------------------
# PARETO SET 1 
# -------------------------------------
# pareto 4 = model | run | gen | ind | 
	tmp = []
	for m in pareto[s][1].iterkeys():		
		tmp.append(np.hstack((np.ones((len(pareto[s][1][m]),1))*models_to_id[m], pareto[s][1][m][:,0:5])))            	
	tmp = np.vstack(tmp)
	tmp = tmp[tmp[:,4].argsort()][::-1]                        
	if len(tmp):
		pareto4[s] = []
		pareto4[s] = [tmp[0]]
		for pair in tmp[1:]:
			if pair[5] >= pareto4[s][-1][5]:
				pareto4[s].append(pair)
		pareto4[s] = np.array(pareto4[s])  

# --------------------------------------
# TCHEBYTCHEV SET 1
# --------------------------------------
	tmp = pareto4[s][:,4:]
	positif = (tmp[:,0]>0)*(tmp[:,1]>0)
	tmp = tmp[positif]
	ideal = np.max(tmp[:,0:2], 0)
	nadir = np.min(tmp[:,0:2], 0)
	value = 0.5*((ideal-tmp)/(ideal-nadir))
	value = np.max(value, 1)+0.001*np.sum(value,1)
	ind_best_point = np.argmin(value)
	# Saving best individual
	best_ind = pareto4[s][ind_best_point]
	m = id_to_models[int(best_ind[0])]
	run_ = int(best_ind[1])
	gen_ = int(best_ind[2])
	num_ = int(best_ind[3])
	data_run = data[s][1][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test_v1[s+str(1)] = dict({m:dict(zip(p_order[m],tmp[4:]))})                        
	
# -----------------------------------
# CHECKING PYTHON MODELS + SAVING RT TIMING for SET 1
# -----------------------------------	
	print "from sferes :", tmp[2]-2000, tmp[3]-500
	timing_v1[s] = dict({})
	model = models_set[m][1]
	model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], p_test_v1[s+str(1)][m], sferes = True)        
	with open("meg/"+s+".pickle", "rb") as f:
		data2 = pickle.load(f)
	opt = EA(data2, s, model)                                
	for i in xrange(opt.n_blocs):
		opt.model.startBloc()
		for j in xrange(opt.n_trials):
			opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
			opt.model.updateValue(opt.responses[i,j])
	opt.fit[0] = float(np.sum(opt.model.value))
	timing_v1[s][m] = [np.median(opt.model.reaction)]
	opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
	timing_v1[s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
	opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
	timing_v1[s][m] = np.array(timing_v1[s][m])
	opt.fit[1] = float(-opt.leastSquares())
	
	print "from test   :", opt.fit[0], opt.fit[1], "\n"

# -----------------------------------
# CLASSIFYING DATA FROM LEAST SQUARE ERROR AFTER TEST = PARETO 5
# -----------------------------------
	nb_blocs = 4
	nb_trials = 48
	nb_repeat = 100
	case = 'meg'
	cats = CATS(nb_trials)
	pareto5[s] = np.zeros((len(pareto3[s]),2)) 
	timing_all[s] = np.zeros((len(pareto3[s]),2))
	for l in xrange(len(pareto3[s])):
		m = id_to_models[int(pareto3[s][l,0])]
		set_ = int(pareto3[s][l,1])		
		run_ = int(pareto3[s][l,2])
		gen_ = int(pareto3[s][l,3])
		num_ = int(pareto3[s][l,4])
		data_run = data[s][set_][m][run_]
		tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]	
		if m in ['selection', 'bayesian', 'qlearning']:
			order = p_order[m]
		elif m in ['mixture', 'fusion']:			
			order = p_order_5[m]		 
		parameter = dict(zip(p_order[m],tmp[4:]))
		# MUST DO A PRE RUN TO COMPUTE TIMING		
		model = models_set[m][set_]
		model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameter, sferes = True)        
		# model.setAllParameters(parameter)		
		opt = EA(data2, s, model)                                
		for i in xrange(opt.n_blocs):			
			model.startBloc()
			for j in xrange(opt.n_trials):
				model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
				model.updateValue(opt.responses[i,j])		
		timing_all[s][l,0] = np.median(opt.model.reaction)
		opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
		timing_all[s][l,1] = np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25)		
		# THE RUN
		model = models_set[m][set_]
		model.__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameter, sferes = False)        
		model.startExp()
		for k in xrange(nb_repeat):		
			for i in xrange(nb_blocs):
				cats.reinitialize()
				cats.stimuli = np.array(map(_convertStimulus, human.subject[case][s][i+1]['sar'][:,0]))[0:nb_trials]
				model.startBloc()
				for j in xrange(nb_trials):                    
					state = cats.getStimulus(j)
					action = model.chooseAction(state)
					reward = cats.getOutcome(state, action, case=case)                
					model.updateValue(reward)		
		rtm = np.array(model.reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
		state = convertStimulus(np.array(model.state)).reshape(nb_repeat, nb_blocs, nb_trials)
		action = np.array(model.action).reshape(nb_repeat, nb_blocs, nb_trials)
		responses = np.array(model.responses).reshape(nb_repeat, nb_blocs, nb_trials)    
		tmp = np.zeros((nb_repeat, 15))

		for i in xrange(nb_repeat):
			rtm[i] = rtm[i] - timing_all[s][l,0]
			rtm[i] = rtm[i] / timing_all[s][l,1]	        
			step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i], case)
			if len(step[5]):
				tmp[i] = computeMeanRepresentativeSteps(step)[0]
			else:
				# QUITE BAD
				step[5] = np.array([0.0])
				tmp[i] = computeMeanRepresentativeSteps(step)[0]
		
		pcr_model = extractStimulusPresentation(responses.reshape(nb_repeat*nb_blocs, nb_trials),
												state.reshape(nb_repeat*nb_blocs, nb_trials),
												action.reshape(nb_repeat*nb_blocs, nb_trials),
												responses.reshape(nb_repeat*nb_blocs, nb_trials))
		pcr_human = extractStimulusPresentation(data2['reward'], data2['state'], data2['action'], data2['reward'])		
		rt_model = tmp.mean(0)
		rt_human = data2['mean'][0]
		pareto5[s][l,0] = np.sum(np.power(pcr_model['mean'].flatten() - pcr_human['mean'].flatten(), 2.0))
		pareto5[s][l,1] = np.sum(np.power(rt_model[1:]-rt_human[1:], 2.0))


	ind_best_point = np.argmin(pareto5[s].sum(1))
	# Saving best individual
	best_ind = pareto3[s][ind_best_point]	
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])
	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	order = p_order_5[m]
	p_test_pre_test[s+str(set_)] = dict({m:dict(zip(order,tmp[4:]))})                        
	position_pre_test[s+str(set_)] = best_ind[5:]
	timing_pre_test[s] = {m:timing_all[s][ind_best_point]}







with open("pareto2.pickle", 'wb') as f:
	pickle.dump(pareto2, f)
with open("pareto3.pickle", 'wb') as f:
	pickle.dump(pareto3, f)
with open("pareto4.pickle", 'wb') as f:
	pickle.dump(pareto4, f)	
with open("pareto5.pickle", 'wb') as f:
	pickle.dump(pareto5, f)
with open("position.pickle", 'wb') as f:
	pickle.dump(position, f)
with open("position_pre_test.pickle", 'wb') as f:
	pickle.dump(position_pre_test, f)	

with open("p_test_last_set.pickle", 'wb') as f:
	pickle.dump(p_test, f)
with open("p_test2_last_set.pickle", 'wb') as f:
	pickle.dump(p_test2, f)
with open("p_test_last_set_v1.pickle", 'wb') as f:
	pickle.dump(p_test_v1, f)
with open("p_test_pre_test.pickle", 'wb') as f:
	pickle.dump(p_test_pre_test, f)
with open("timing.pickle", 'wb') as f:
	pickle.dump(timing, f)
with open("timing_v1.pickle", 'wb') as f:
	pickle.dump(timing_v1, f)
with open("timing_all.pickle", 'wb') as f:
	pickle.dump(timing_all, f)
with open("timing_pre_test.pickle", 'wb') as f:
	pickle.dump(timing_pre_test, f)
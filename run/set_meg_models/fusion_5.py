#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Parameters for sferes optimization 
# To speed up the process and avoid list
# n_trials = 39
n_trials = 48
n_blocs = 4

def SoftMaxValues(values, beta):
	tmp0 = values - np.max(values)
	tmp = np.exp(tmp0*float(beta))
	return  tmp/float(np.sum(tmp))
def convertStimulus(state):
    return (state == 's1')*1+(state == 's2')*2 + (state == 's3')*3
def convertAction(action):
    return (action=='thumb')*1+(action=='fore')*2+(action=='midd')*3+(action=='ring')*4+(action=='little')*5




class fusion_4():
	""" fusion strategy
	
	"""
	def __init__(self, states, actions, parameters={"length":1}, sferes = False):
		#State Action Spaces
		self.states=states
		self.actions=actions
		#Parameters
		self.sferes = sferes
		self.parameters = parameters
		self.n_action = int(len(actions))
		self.n_state = int(len(states))
		self.bounds = dict({"beta":[0.0, 100.0], # temperature for final decision                            
							'alpha':[0.0, 1.0],
							"length":[1, 10],
							"threshold":[0.00001, 1000.0], # sigmoide parameter
							"noise":[0.0, 0.1],
							"gain":[0.00001, 10000.0], # sigmoide parameter 
							"sigma":[0.0, 20.0],
							"gamma":[0.0, 100.0],
							"kappa":[0.0, 1.0],
							"shift":[-20.0, 0.0],
							"xi":[0.0, 20.0]
							}) 
							

		#Probability Initialization
		self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
		self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
		self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
		self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
		self.p_a_mf = None
		self.p_a_mb = None
		self.p = None
		self.p_a = None
		self.pA = None
		# QValues model free
		self.values_mf = np.zeros((self.n_state, self.n_action))
		self.q_values = None
		# Control initialization
		self.nb_inferences = 0
		self.n_element= 0
		self.current_state = None
		self.current_action = None
		self.max_entropy = -np.log2(1./self.n_action)
		self.Hb = self.max_entropy
		self.Hf = self.max_entropy
		self.N = 0        
		self.p_a = np.zeros(self.n_action)
		self.delta = 0
		# List Init
		if self.sferes:
			self.value = np.zeros((n_blocs, n_trials))
			self.reaction = np.zeros((n_blocs, n_trials))
			self.p_decision = None
			self.p_retrieval = None
		else:
			self.state = list()
			self.action = list()
			self.responses = list()
			self.reaction = list()
			self.value = list()
			self.pdf = list()
			self.Hall = list()
			self.update = list()
			# prediction
			self.h_bayes_only = 0.0
			self.h_ql_only = 0.0

	def setParameters(self, name, value):            
		if value < self.bounds[name][0]:
			self.parameters[name] = self.bounds[name][0]
		elif value > self.bounds[name][1]:
			self.parameters[name] = self.bounds[name][1]
		else:
			self.parameters[name] = value

	def setAllParameters(self, parameters):
		for i in parameters.iterkeys():
			if i in self.bounds.keys():
				self.setParameters(i, parameters[i])

	def startBloc(self):
		if not self.sferes:
			self.state.append([])
			self.action.append([])
			self.responses.append([])
			self.reaction.append([])
			self.pdf.append([])
			self.Hall.append([])
			self.update.append([])
		self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
		self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
		self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
		self.values_mf = np.zeros((self.n_state, self.n_action))
		self.nb_inferences = 0
		self.n_element = 0
		self.current_state = None
		self.current_action = None
		self.Hb = self.max_entropy
		self.Hf = self.max_entropy    
		self.p_decision = None
		self.p_retrieval = None
		self.Hb = self.max_entropy
		self.Hf = self.max_entropy
		self.N = 0        
		self.p_a = np.zeros(self.n_action)
		self.delta = 0		

	def startExp(self):        
		self.state = list()
		self.action = list()
		self.responses = list()
		self.reaction = list()
		self.value = list()        
		self.pdf = list()
		self.Hall = list()
		self.update = list()

	def sample(self, values):
		tmp = [np.sum(values[0:i]) for i in range(len(values))]
		return np.sum(np.array(tmp) < np.random.rand())-1

	def inferenceModule(self):        
		tmp = self.p_a_s[self.nb_inferences] * np.vstack(self.p_s[self.nb_inferences])
		self.p = self.p + self.p_r_as[self.nb_inferences] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[self.nb_inferences].shape)
		self.nb_inferences+=1

	def evaluationModule(self):
		tmp = self.p/np.sum(self.p)
		p_ra_s = tmp[self.current_state]/np.sum(tmp[self.current_state])
		p_r_s = np.sum(p_ra_s, axis = 0)
		p_a_rs = p_ra_s/p_r_s		
		self.p_a_mb = p_a_rs[:,1]/p_a_rs[:,0]		
		p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
		self.Hb = -np.sum(p_a_mb*np.log2(p_a_mb))		

	def sigmoideModule(self):
		np.seterr(invalid='ignore')
		x = 2*self.max_entropy-self.Hb-self.Hf
		self.pA = 1/(1+((self.n_element-self.nb_inferences)**self.parameters['threshold'])*np.exp(-x*self.parameters['gain']))
		# print "n=",self.n_element," i=", self.nb_inferences, " Hb=", self.Hb, " Hf=", self.Hf, " x=", x, " p(A)=",self.pA, "threshold= ", self.parameters['threshold'], "gain = ", self.parameters['gain']
		return np.random.uniform(0,1) > self.pA
	
	def fusionModule(self):
		np.seterr(invalid='ignore')
		self.values_net = self.p_a_mb+self.values_mf[self.current_state]
		tmp = SoftMaxValues(self.values_net, float(self.parameters['beta']))
		
		ninf = np.isinf(tmp).sum()        

		if ninf:
			print "INF"
		if np.isinf(tmp).sum():            
			self.p_a = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
		else :
			self.p_a = tmp/np.sum(tmp)   
		
		if np.sum(self.p_a == 0.0):
			self.p_a+=1e-8;
			self.p_a = self.p_a/self.p_a.sum()  

		if not self.sferes:
			# qlearning
			tmp = np.exp(self.values_mf[self.current_state]*float(self.parameters['beta']))
			pa = tmp/np.sum(tmp)
			# print "Q_ql("+str(self.current_state)+")=",self.values_mf[self.current_state]            
			# print "p_ql("+str(self.current_state)+")=",pa            
			self.h_ql_only = -np.sum(pa*np.log2(pa))
			# print "H_ql =", self.h_ql_only
			# bayesian
			tmp = np.exp(self.p_a_mb*float(self.parameters['beta']))
			pa = tmp/np.sum(tmp)
			# print "N=", self.nb_inferences
			# print "Q_ba("+str(self.current_state)+")=",self.p_a_mb
			# print "p_ba("+str(self.current_state)+")=",pa
			self.h_bayes_only = -np.sum(pa*np.log2(pa))
			# print "H_ba =", self.h_bayes_only			
			 
	def computeValue(self, s, a, ind):
		self.current_state = s
		self.current_action = a		
		self.p = self.uniform[:,:,:]
		self.Hb = self.max_entropy
		self.p_a_mf = SoftMaxValues(self.values_mf[self.current_state], self.parameters['gamma'])
		self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
		self.nb_inferences = 0
		self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        
		self.p_decision = np.zeros(int(self.parameters['length'])+1)
		self.p_retrieval= np.zeros(int(self.parameters['length'])+1)
		self.p_sigmoide = np.zeros(int(self.parameters['length'])+1)
		self.p_actions = np.zeros((int(self.parameters['length'])+1,self.n_action))
		self.evolution_entropy = np.zeros(int(self.parameters['length'])+1)
		self.q_values = np.zeros((int(self.parameters['length'])+1, self.n_action))
		reaction = np.zeros(int(self.parameters['length'])+1)
		# START
		self.sigmoideModule()
		self.p_sigmoide[0] = self.pA
		self.p_decision[0] = self.pA
		self.p_retrieval[0] = 1.0-self.pA
		self.evolution_entropy[0] = self.Hb
		self.fusionModule()
		self.q_values[0] = self.values_net
		self.p_actions[0] = self.p_a
		H = -(self.p_a*np.log2(self.p_a)).sum()
		reaction[0] = float(H)
		# reaction[0] = float(H)
		for i in xrange(self.n_element):            
			self.inferenceModule()
			self.evaluationModule()
			self.fusionModule()
			self.p_actions[i+1] = self.p_a
			self.q_values[i+1] = self.values_net
			H = -(self.p_a*np.log2(self.p_a)).sum()
			N = self.nb_inferences+1.0
			reaction[i+1] = float(((np.log2(N))**self.parameters['sigma'])+H)
			self.sigmoideModule()
			############
			self.p_sigmoide[i+1] = self.pA            
			self.p_decision[i+1] = self.pA*self.p_retrieval[i]            
			self.p_retrieval[i+1] = (1.0-self.pA)*self.p_retrieval[i]        
			self.evolution_entropy[i+1] = self.Hb
			############

		self.N = self.nb_inferences
		self.p_a = np.dot(self.p_decision,self.p_actions)		
		self.value[ind] = float(np.log(self.p_a[self.current_action]))        
		# print self.value[ind]
		self.reaction[ind] = float(np.sum(reaction*self.p_decision.flatten()))    
		
			
	def chooseAction(self, state):
		self.state[-1].append(state)
		self.current_state = convertStimulus(state)-1
		self.p = self.uniform[:,:,:]
		self.Hb = self.max_entropy
		self.p_a_mf = SoftMaxValues(self.values_mf[self.current_state], self.parameters['gamma'])
		self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
		self.nb_inferences = 0
		self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
		
		while self.sigmoideModule():			
			self.inferenceModule()
			self.evaluationModule()

		self.fusionModule()
		self.current_action = self.sample(self.p_a)
		self.value.append(float(self.p_a[self.current_action]))
		self.action[-1].append(self.current_action)                
		self.Hall[-1].append([float(self.Hb), float(self.Hf)])
		H = -(self.p_a*np.log2(self.p_a)).sum()
		self.Hl = H

		# print "mf ", self.values_mf[self.current_state]
		# print "mb ", self.p_a_mb
		# print self.values_net
		# print self.p_a
		# print "s = ", self.current_state, "a = ", self.current_action
		
		N = float(self.nb_inferences+1)        				
		self.reaction[-1].append(float(((np.log2(N))**self.parameters['sigma'])+H))
		self.pdf[-1].append(N)

		# self.reaction[-1].append(N-1)
		
		return self.actions[self.current_action]

	def updateValue(self, reward):				
		# Updating model free
		r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0                
		self.delta = float(r)-self.values_mf[self.current_state, self.current_action]        
		self.values_mf[self.current_state, self.current_action] = self.values_mf[self.current_state, self.current_action]+self.parameters['alpha']*self.delta        
		index = range(self.n_action)
		index.pop(int(self.current_action))        
		self.values_mf[self.current_state][index] = self.values_mf[self.current_state][index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[self.current_state][index])            
		r = (reward==0)*0.0+(reward==1)*1.0+(reward==-1)*0.0                
		if not self.sferes:
			self.responses[-1].append(r)		
		if self.delta < self.parameters['shift'] or self.delta > self.parameters['xi']:			
			if not self.sferes:
				self.update[-1].append(1.0)
			if self.parameters['noise']:
				self.p_s = self.p_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_state*np.ones(self.p_s.shape))
				self.p_a_s = self.p_a_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_action*np.ones(self.p_a_s.shape))
				self.p_r_as = self.p_r_as*(1-self.parameters['noise'])+self.parameters['noise']*(0.5*np.ones(self.p_r_as.shape))
			#Shifting memory            
			if self.n_element < int(self.parameters['length']):
				self.n_element+=1
			self.p_s[1:self.n_element] = self.p_s[0:self.n_element-1]
			self.p_a_s[1:self.n_element] = self.p_a_s[0:self.n_element-1]
			self.p_r_as[1:self.n_element] = self.p_r_as[0:self.n_element-1]
			self.p_s[0] = 0.0
			self.p_a_s[0] = np.ones((self.n_state, self.n_action))*(1/float(self.n_action))
			self.p_r_as[0] = np.ones((self.n_state, self.n_action, 2))*0.5
			#Adding last choice                 
			self.p_s[0, self.current_state] = 1.0        
			self.p_a_s[0, self.current_state] = 0.0
			self.p_a_s[0, self.current_state, self.current_action] = 1.0
			self.p_r_as[0, self.current_state, self.current_action] = 0.0
			self.p_r_as[0, self.current_state, self.current_action, int(r)] = 1.0        
		else:
			if not self.sferes:
				self.update[-1].append(0.0)

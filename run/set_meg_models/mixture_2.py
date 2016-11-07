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




class CSelection():
    """Class that implement Collins models for action selection
    Model-based must be provided
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, states, actions, parameters={'length':1, 'weight':0.5}, sferes = False):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.sferes = sferes
        self.parameters = parameters
        self.n_action=int(len(actions))
        self.n_state=int(len(states))
        self.max_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[1, 10], 
                            "threshold":[0.01, self.max_entropy], 
                            "noise":[0.0, 0.1],                            
                            'alpha':[0.0, 1.0],
                            "beta":[0.0, 100.0], # QLEARNING
                            "sigma":[0.0, 20.0], 
                            "weight":[0.0, 1.0],
                            "kappa":[0.0, 1.0],
                            "shift":[-20.0, 20.0]})
                            
                            

        # Probability Initialization        
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)    
        self.p = None        
        self.p_a = None
        # Specific to collins        
        self.w = np.ones(self.n_state)*self.parameters['weight']
        self.q_mb = np.zeros((self.n_action))
        # Q-values model free
        self.q_mf = np.zeros((self.n_state, self.n_action))
        self.p_a_mf = None
        # Various Init
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None        
        self.entropy = self.max_entropy        
        self.n_element = 0
        self.q_values = np.zeros(self.n_action)
        self.Hb = 0.0
        self.Hf = 0.0
        self.N = 0
        self.delta = 0.0        
        # Optimization init
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_r_s = np.ones(2)*0.5
        #List Init
        if self.sferes:
            self.value = np.zeros((n_blocs, n_trials))
            self.reaction = np.zeros((n_blocs, n_trials))
        else:
            self.state=list()        
            self.action=list()
            self.responses=list()        
            self.reaction=list()
            self.value=list()
            self.pdf = list()
            self.Hall = list()
            self.pdf = list()


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
            self.weights.append([])
            self.p_wm.append([])
            self.p_rl.append([])
            self.Hall.append([])
            self.pdf.append([])
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
        self.w = np.ones(self.n_state)*self.parameters['weight']
        self.q_mb = np.zeros((self.n_action))
        self.q_mf = np.zeros((self.n_state, self.n_action))
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)    
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None

    def startExp(self):
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.state=list()
        self.action=list()
        self.reaction=list()
        self.responses=list()
        self.value=list()
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
        self.weights=list()
        self.p_wm=list()
        self.p_rl=list()
        self.Hall=list()
        self.pdf=list()

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
        self.q_mb = p_a_rs[:,1]/p_a_rs[:,0]        
        # self.p_a_mb = np.exp(self.q_mb*float(self.parameters['gain']))        
        self.p_a_mb = self.q_mb/np.sum(self.q_mb)
        # self.p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)        
        self.Hb = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))
        

    def fusionModule(self):
        np.seterr(invalid='ignore')
        self.p_a_mf = np.exp(self.q_mf[self.current_state]*float(self.parameters['beta']))
        self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)
        self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
        self.p_a = (1.0-self.w[self.current_state])*self.p_a_mf + self.w[self.current_state]*self.p_a_mb                
        self.q_values = self.p_a      

        #nombre de inf
        ninf = np.isinf(self.p_a).sum()  
        if np.isinf(self.p_a).sum():
            self.p_a = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
        else :
            self.p_a = self.p_a/np.sum(self.p_a)   
        
        if not self.sferes:
            # qlearning
            tmp = np.exp(self.q_mf[self.current_state]*float(self.parameters['beta']))
            pa = tmp/np.sum(tmp)
            self.h_ql_only = -np.sum(pa*np.log2(pa))
            # bayesian            
            self.h_bayes_only = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))

                    
    def updateWeight(self, r):
        if r:
            p_wmc = self.p_a_mb[self.current_action]
            p_rl = self.p_a_mf[self.current_action]
        else:
            p_wmc = 1.0 - self.p_a_mb[self.current_action]
            p_rl = 1.0 - self.p_a_mf[self.current_action]
        self.w[self.current_state] = (p_wmc*self.w[self.current_state])/(p_wmc*self.w[self.current_state] + p_rl * (1.0 - self.w[self.current_state]))
        # self.p_wm[-1].append(self.p_a_mb[self.current_action])
        # self.p_rl[-1].append(self.p_a_mf[self.current_action])
        
    def computeValue(self, s, a, ind):
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.nb_inferences = 0     
        self.evolution_entropy = np.zeros(int(self.parameters['length'])+1)
        self.evolution_entropy[0] = self.Hb        

        while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
            self.inferenceModule()
            self.evaluationModule()            
            self.evolution_entropy[self.nb_inferences] = self.Hb
        
        self.fusionModule()
        # print ind, self.p_a
        H = -(self.p_a*np.log2(self.p_a)).sum()
        self.N = self.nb_inferences
        # if np.isnan(H): H = 0.005

        self.value[ind] = float(np.log(self.p_a[self.current_action]))
        self.reaction[ind] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)        

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.nb_inferences = 0             
        # print self.Hb, self.parameters['threshold'], self.nb_inferences, self.n_element
        while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()
        
        self.fusionModule()        
        self.current_action = self.sample(self.p_a)
        self.value.append(float(self.p_a[self.current_action]))
        self.action[-1].append(self.current_action)
        self.weights[-1].append(self.w[self.current_state])
        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        self.Hl = H
        self.reaction[-1].append(float((np.log2(N)**self.parameters['sigma'])+H))        
        self.Hall[-1].append([float(self.Hb), float(self.Hf)])
        self.pdf[-1].append(N)
        return self.actions[self.current_action]

    def updateValue(self, reward):
        r = int((reward==1)*1)
        if not self.sferes:
            self.responses[-1].append(r)        
        # Specific to Collins model
        self.updateWeight(float(r))
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
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        # delta = float(r)+self.parameters['gamma']*np.max(self.q_mf[self.current_state])-self.q_mf[self.current_state, self.current_action]                
        self.delta = float(r)-self.q_mf[self.current_state, self.current_action]                        
        self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['alpha']*self.delta
        # if r>0:        
        #     self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['alpha']*delta
        # elif r<=0:
        #     self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['omega']*delta                    

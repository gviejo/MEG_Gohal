#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#from prettytable import PrettyTable
from copy import deepcopy
import cPickle as pickle
import sys
from scipy import stats

#from scipy.stats import chi2_contingency, sem
from scipy.stats import sem


# def displayQValues(states, actions, values, ind = 0):
#     foo = PrettyTable()
#     line_actions = ["State\Action"]
#     for i in actions:
#         line_actions.append(i)
#     foo.set_field_names(line_actions)
#     for j in states:
#         line = [j]
#         for v in actions:
#             line.append(round(values[ind][values[(j,v)]], 3))
#         foo.add_row(line)
#     foo.printt()

def print_dict(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '%s%s%s%s' %(ident,braces*'[',key,braces*']') 
            print_dict(value, ident+'  ', braces+1)
        else:
            print ident+'%s = %s' %(key, value)


def createQValuesDict(states, actions):
    #return a big dict
    values = {0:np.zeros(len(states)*len(actions))}
    tmp = 0
    for s in states:
        values[s] = []
        n = 0
        for a in actions:
            values[(s,a)] = tmp
            values[s].append(tmp)
            values[(s,n)] = a
            tmp = tmp + 1
            n = n + 1
    return values

def createRewardRateDict():
    return dict({'rate':0,
                 'reward':0,
                 'tau':0,
                 'step':0,
                 'tau_list':[0],
                 'r_list':[0],
                 ('s0', 'pl'):8,
                 ('s0', 'em'):12,
                 ('s1', 'pl'):12,
                 ('s1', 'em'):0,
                 'r*tau':[]})

def createCovarianceDict(n, init_cov, eta):
    #cov = np.eye(n)*init_cov+(1-np.eye(n,n))*((1-init_cov)/(n-1))    
    cov = np.eye(n)
    return dict({'cov':cov,
                 'noise':np.eye(n,n)*init_cov*eta})


def getBestAction(state, values, ind = 0):
    return values[(state, np.where(values[ind][values[state]] == np.max(values[ind][values[state]]))[0][0])]

def getBestActionSoftMax(state, values, beta, ind = 0):
    tmp = np.exp(values[ind][values[state]]*float(beta))
    tmp = tmp/float(np.sum(tmp))
    tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
    return values[(state, np.sum(np.array(tmp) < np.random.rand())-1)]

def SoftMax(values, beta):
    tmp = np.exp(values*float(beta))
    tmp = tmp/float(np.sum(tmp))
    tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
    return np.sum(np.array(tmp) < np.random.rand())-1

def SoftMaxValues(values, beta):
    tmp = np.exp(values*float(beta))
    return  tmp/float(np.sum(tmp))

def computeEntropy(values, beta):
    if np.sum(values) == 1 and np.min(values) > 0.0:
        return -np.sum(values*np.log2(values))
    else:
        tmp = np.exp(values*float(beta))
        tmp = tmp/float(np.sum(tmp))
        return -np.sum(tmp*np.log2(tmp))


def computeVPIValues(mean, variance):
    #WARNING input and output very specific
    # mean = array(current state), variance = array(current state)
    #vpi = array(current state)
    vpi = np.zeros((len(mean)))
    ind = np.argsort(mean)
    vpi[ind[-1]] = (mean[ind[-2]]-mean[ind[-1]])*stats.norm.cdf(mean[ind[-2]], mean[ind[-1]], np.sqrt(variance[ind[-1]])) + (np.sqrt(variance[ind[-1]])/np.sqrt(2*np.pi))*np.exp(-(mean[ind[-2]]-mean[ind[-1]])**2/(2*variance[ind[-1]]))
    for i in range(len(mean)-2, -1, -1):
        vpi[ind[i]] = (mean[ind[i]]-mean[ind[-1]])*(1-stats.norm.cdf(mean[ind[-1]], mean[ind[i]], np.sqrt(variance[ind[i]]))) + (np.sqrt(variance[ind[i]])/np.sqrt(2*np.pi))*np.exp(-(mean[ind[-1]]-mean[ind[i]])**2/(2*variance[ind[i]]))        
    return vpi
       
def updateRewardRate(reward_rate, sigma, delay = 0.0):
    return ((1-sigma)**(1+delay))*reward_rate['rate']+sigma*reward_rate['reward']

def updateQValuesHabitual(values, delta, alpha):
    return values+delta*alpha

def computeSigmaPoints(values, covariance, kappa=0.5):
    n = len(values)
    point = np.zeros((2*n+1,n))
    point[0] = values
    c = np.linalg.cholesky((n+kappa)*covariance)
    point[range(1,n+1)] = values+np.transpose(c)
    point[range(n+1, 2*n+1)] = values-np.transpose(c)
    weights = np.zeros((2*n+1,1))
    weights[0] = kappa/(n+kappa)
    weights[1:2*n+1] = 1/(2*n+kappa)
    return point, weights

def computeExplorationCost(reward_rate, tau, transition):
    return tau*reward_rate[transition]
    
def createTransitionDict(state1, action, state2, stopstate):
    transition = dict()
    n = float(len(set(state1)))
    transition[None] = stopstate
    for i,j,k in zip(state1, action, state2):
        transition[(i,j,k)] = 1/n
        transition[(i,j)] = k
    return transition

def updateTransition(transition_values, transition, phi):
    transition_values[transition] = (1-phi)*transition_values[transition] + phi
    for i in transition_values.iterkeys():
        if i <> transition and i <> None and len(i) == 3 and i[0] == transition[0]:
            transition_values[i] = (1-phi)*transition_values[i]
    return transition_values

def updateRewardsFunction(rewards_function, state, action, rau):
    rewards_function[1] = (1-rau)*rewards_function[1]+rau*rewards_function[0][rewards_function[(state, action)]]
    return rewards_function

def computeGoalValue(values, state, action, rewards, gamma, depth, phi, rau):
    rewards_function = deepcopy(rewards)
    rewards_function[1] = rewards_function[0].copy()
    rewards_function = updateRewardsFunction(rewards_function, state, action, rau)
    transition = createTransitionDict(['s0','s0','s1','s1'],['pl','em','pl','em'],['s1','s0','s0',None], 's0') #<====VERY BAD==============    NEXT_STATE = TRANSITION[(STATE, ACTION)]
    next_state = transition[(state, action)]
    if next_state == None:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[transition[None]]])        
    else:
        transition = updateTransition(transition, (state, action, next_state), phi)
        tmp = np.max([computeGoalValueRecursive(values, next_state, a, rewards_function.copy(), transition.copy(), gamma, depth-1, phi, rau) for a in values[next_state]])
        value = rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*tmp
        return value

def computeGoalValueRecursive(values, state, a, rewards_function, transition, gamma, depth, phi, rau):
    action = values[(state, values[state].index(a))]
    next_state = transition[(state, action)]
    rewards_function = updateRewardsFunction(rewards_function, state, action, rau)
    transition = updateTransition(transition, (state, action, next_state), phi)
    if next_state == None:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[transition[None]]])        
    elif depth == 0:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[next_state]])
    else:
        tmp = np.max([computeGoalValueRecursive(values, next_state, a, rewards_function.copy(), transition.copy(), gamma, depth-1, phi, rau) for a in values[next_state]])
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*tmp


def testQValues(states, values, beta, ind, niter):
    tmp = np.zeros((4)) #VBAAAAAAD
    for i in range(niter):
        for s in states:
            a = getBestActionSoftMax(s, values, beta, ind)
            tmp[values[(s,a)]] = tmp[values[(s, a)]] + 1 
    tmp = tmp/float(niter)
    return tmp
                

def saveData(file_name, data):
    f = open(file_name, 'w')
    pickle.dump(data, f, protocol = 2)
    f.close()

def loadData(file_name):
    f = open(file_name, 'r')
    data = pickle.load(f)
    f.close()
    return data
        
def searchStimOrder(s, a, r):
    assert len(s) == len(a) == len(r)
    incorrect = dict()
    tmp = np.zeros(3)
    st = np.array([1,2,3])
    for j in st:
        if len(np.where((r == 1) & (s == j))[0]):
            correct = np.where((r == 1) & (s == j))[0][0]
            t = len(np.where((r[0:correct] == 0) & (s[0:correct] == j))[0])
            incorrect[t] = j
            tmp[np.where(st == j)[0][0]] = t
    if np.sum(tmp == 0) == 2:
        return tuple((st[tmp != 0][0], st[tmp == 0][0], st[tmp == 0][1]))
    if np.sum(tmp == 0) == 1 and len(np.unique(tmp)) == 3:
        return tuple((st[tmp == np.min(tmp[tmp!=0])][0], st[tmp == np.max(tmp[tmp!=0])][0] ,st[tmp == 0][0]))
    elif np.sum(tmp == 0) == 1 and len(np.unique(tmp)) == 2:
        return tuple((st[tmp != 0][0], st[tmp != 0][1], st[tmp == 0][0]))
    elif np.sum(tmp == np.min(tmp)) == 1 and len(np.unique(tmp)) == 2:
        #find the first one who got the solution        
        first = st[tmp == np.min(tmp)][0]
        rest = st[tmp != np.min(tmp)]
        if np.where((s == rest[0]) & (r == 1))[0][0] < np.where((s == rest[1]) & (r == 1))[0][0]:
            return tuple((first, rest[0], rest[1]))
        else:       
            return tuple((first, rest[1], rest[0]))  
    elif np.sum(tmp == np.min(tmp)) == 2 and len(np.unique(tmp)) == 2:        
        if np.where(s == st[tmp == np.min(tmp)][0])[0][0] < np.where(s == st[tmp == np.min(tmp)][1])[0][0]:            
            return tuple((st[tmp == np.min(tmp)][0], st[tmp == np.min(tmp)][1], st[tmp == np.max(tmp)][0]))
        else:            
            return tuple((st[tmp == np.min(tmp)][1], st[tmp == np.min(tmp)][0], st[tmp == np.max(tmp)][0]))
    elif len(np.unique(tmp)) == 3:
        return tuple(st[np.argsort(tmp)])        


def computeMeanReactionTime(data, case = None, ind = 40):
    tmp = []
    if case.lower() == 'fmri':
        for s in data.iterkeys():
            for b in data[s].iterkeys():
                tmp.append(data[s][b]['RT'].flatten()[0:ind])
    #TODO for meg
    reaction = dict({'mean':np.mean(tmp, 0),
                     'sem':stats.sem(tmp, 0)})
    return reaction
                
def getRepresentativeSteps(data, stimulus, action, responses, case = 'fmri', pastouche=16):
    m, n = data.shape
    bad_trials = 0
    incorrect_trials = 0
    assert(data.shape == stimulus.shape == responses.shape)    
    indice = np.zeros((m,n))
    if case == 'meg':
        for i in xrange(m):
            first, second, third = searchStimOrder(stimulus[i], action[i], responses[i])        
            # first
            first_correct_position = np.where((stimulus[i] == first) & (responses[i] == 1))[0][0]        
            indice[i, first_correct_position] = 2
            #second_correct_position = np.where((stimulus[i] == first) & (responses[i] == 1))[0][1]
            #indice[i, second_correct_position] = 6
            #first_wrong_position = np.where((stimulus[i, 0:first_correct_position] == first) & (responses[i,0:first_correct_position] == 0))[0][0]
            #indice[i, first_wrong_position] = 1
            first_wrong_position = np.where((stimulus[i,0:first_correct_position] == first) & (responses[i,0:first_correct_position] == 0))[0]
            indice[i][first_wrong_position] = 1
            second_position = np.where((stimulus[i] == first)&(indice[i] == 0))[0][0]
            indice[i][second_position] = 6
            tmp = 7
            for j in np.where((stimulus[i] == first) & (indice[i] == 0))[0]:
                    indice[i,j] = tmp; tmp += 1

            #second 
            if len(np.where((stimulus[i] == second) & (responses[i] == 1))[0]):
                first_correct_position = np.where((stimulus[i] == second) & (responses[i] == 1))[0][0]        
                indice[i, first_correct_position] = 4
                #second_correct_position = np.where((stimulus[i] == second) & (responses[i] == 1))[0][1]
                #indice[i, second_correct_position] = 6
                wrong_positions = np.where((stimulus[i,0:first_correct_position] == second) & (responses[i, 0:first_correct_position] == 0))[0]
                first_wrong_action = action[i][wrong_positions[0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == first_wrong_action))[0]] = 1
                if len(np.where((stimulus[i, 0:first_correct_position] == second) & (action[i, 0:first_correct_position] != first_wrong_action))[0]):
                    second_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == second)             
                                                            & (action[i, 0:first_correct_position] != first_wrong_action))[0][0]]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == second_wrong_action))[0]] = 2
                    third_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == second) 
                                                            & (action[i, 0:first_correct_position] != first_wrong_action)
                                                            & (action[i, 0:first_correct_position] != second_wrong_action))[0][0]]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == third_wrong_action))[0]] = 3
                    if len(np.where((stimulus[i] == second)&(indice[i] == 0))[0]):  
                        second_position = np.where((stimulus[i] == second)&(indice[i] == 0))[0][0]
                        indice[i][second_position] = 6
                        tmp = 7
                        for j in np.where((stimulus[i] == second) & (indice[i] == 0))[0]:
                                indice[i,j] = tmp; tmp += 1

            #third
            if len(np.where((stimulus[i] == third) & (responses[i] == 1))[0]):
                first_correct_position = np.where((stimulus[i] == third) & (responses[i] == 1))[0][0]
                indice[i, first_correct_position] = 5
                wrong_positions = np.where((stimulus[i,0:first_correct_position] == third) & (responses[i, 0:first_correct_position] == 0))[0]
                first_wrong_action = action[i][wrong_positions[0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == first_wrong_action))[0]] = 1
                second_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == third)             
                                                        & (action[i, 0:first_correct_position] != first_wrong_action))[0][0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == second_wrong_action))[0]] = 2
                third_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == third) 
                                                        & (action[i, 0:first_correct_position] != first_wrong_action)
                                                        & (action[i, 0:first_correct_position] != second_wrong_action))[0][0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == third_wrong_action))[0]] = 3
                if len(np.unique(action[i, np.where((stimulus[i,0:first_correct_position] == third))])) == 4:
                    fourth_wrong_action = list(set([1,2,3,4,5]) - set([first_wrong_action, second_wrong_action, third_wrong_action, action[i, first_correct_position]]))[0]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == fourth_wrong_action))[0]] = 4
                if len(np.where((stimulus[i] == third)&(indice[i] == 0))[0]):
                    second_position = np.where((stimulus[i] == third)&(indice[i] == 0))[0][0]
                    indice[i][second_position] = 6
                    tmp = 7
                    for j in np.where((stimulus[i] == third) & (indice[i] == 0))[0]:
                        indice[i,j] = tmp; tmp += 1
    elif case == 'fmri':
        for i in xrange(m):            
            first, second, third = (1,2,3)
            # first
            if len(np.where((stimulus[i] == first) & (responses[i] == 1))[0]):
                first_correct_position = np.where((stimulus[i] == first) & (responses[i] == 1))[0][0]        
                indice[i, first_correct_position] = 2
                first_wrong_position = np.where((stimulus[i,0:first_correct_position] == first) & (responses[i,0:first_correct_position] == 0))[0]
                indice[i][first_wrong_position] = 1
                second_position = np.where((stimulus[i] == first)&(indice[i] == 0))[0][0]
                indice[i][second_position] = 6
                tmp = 7
                for j in np.where((stimulus[i] == first) & (indice[i] == 0))[0]:
                        indice[i,j] = tmp; tmp += 1
            #second 
            if len(np.where((stimulus[i] == second) & (responses[i] == 1))[0]):
                first_correct_position = np.where((stimulus[i] == second) & (responses[i] == 1))[0][0]        
                indice[i, first_correct_position] = 4
                wrong_positions = np.where((stimulus[i,0:first_correct_position] == second) & (responses[i, 0:first_correct_position] == 0))[0]
                first_wrong_action = action[i][wrong_positions[0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == first_wrong_action))[0]] = 1
                if len(np.where((stimulus[i, 0:first_correct_position] == second) & (action[i, 0:first_correct_position] != first_wrong_action))[0]):
                    second_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == second)             
                                                            & (action[i, 0:first_correct_position] != first_wrong_action))[0][0]]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == second_wrong_action))[0]] = 2
                    third_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == second) 
                                                            & (action[i, 0:first_correct_position] != first_wrong_action)
                                                            & (action[i, 0:first_correct_position] != second_wrong_action))[0][0]]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == second) & (action[i,0:first_correct_position] == third_wrong_action))[0]] = 3
                    if len(np.where((stimulus[i] == second)&(indice[i] == 0))[0]):  
                        second_position = np.where((stimulus[i] == second)&(indice[i] == 0))[0][0]
                        indice[i][second_position] = 6
                        tmp = 7
                        for j in np.where((stimulus[i] == second) & (indice[i] == 0))[0]:
                                indice[i,j] = tmp; tmp += 1            
            # third
            if len(np.where((stimulus[i] == third) & (responses[i] == 1))[0]):
                first_correct_position = np.where((stimulus[i] == third) & (responses[i] == 1))[0][0]
                indice[i, first_correct_position] = 5
                wrong_positions = np.where((stimulus[i,0:first_correct_position] == third) & (responses[i, 0:first_correct_position] == 0))[0]
                first_wrong_action = action[i][wrong_positions[0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == first_wrong_action))[0]] = 1
                second_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == third)             
                                                        & (action[i, 0:first_correct_position] != first_wrong_action))[0][0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == second_wrong_action))[0]] = 2
                third_wrong_action = action[i][np.where((stimulus[i, 0:first_correct_position] == third) 
                                                        & (action[i, 0:first_correct_position] != first_wrong_action)
                                                        & (action[i, 0:first_correct_position] != second_wrong_action))[0][0]]
                indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == third_wrong_action))[0]] = 3
                if len(np.unique(action[i, np.where((stimulus[i,0:first_correct_position] == third))])) == 4:
                    fourth_wrong_action = list(set([1,2,3,4,5]) - set([first_wrong_action, second_wrong_action, third_wrong_action, action[i, first_correct_position]]))[0]
                    indice[i, np.where((stimulus[i,0:first_correct_position] == third) & (action[i,0:first_correct_position] == fourth_wrong_action))[0]] = 4
                if len(np.where((stimulus[i] == third)&(indice[i] == 0))[0]):
                    second_position = np.where((stimulus[i] == third)&(indice[i] == 0))[0][0]
                    indice[i][second_position] = 6
                    tmp = 7
                    for j in np.where((stimulus[i] == third) & (indice[i] == 0))[0]:
                        indice[i,j] = tmp; tmp += 1           
    steps = dict()
    for i in range(1,pastouche):
        steps[i] = data[indice == i]
    # if incorrect_trials:
    #     print "Number of incorrect trials : %.2f" % (incorrect_trials/float(m)) + " %"
    # if bad_trials:
    #     print "Number of bad trials : %.2f" % (bad_trials/float(m)) + " %"

    return steps, indice

 

def computeMeanRepresentativeSteps(data):
    assert(type(data) == dict)
    m = []
    s = []
    for i in data.iterkeys():
        if len(data[i]):
            m.append(np.mean(data[i]))
            s.append(sem(data[i]))
            #s.append(np.var(data[i]))
                    
    return np.array(m), np.array(s)
            
    
def extractStimulusPresentation(data, stimulus, action, responses):
    """ Return mean and sem of probabilty of correct responses
    """
    tmp = dict({1:[],2:[],3:[]})
    m, n = responses.shape
    assert(stimulus.shape == responses.shape == action.shape == data.shape)
    bad_trials = 0
    incorrect_trials = 0

    for i in xrange(m):        
        first, second, third = searchStimOrder(stimulus[i], action[i], responses[i])
        tmp[1].append(data[i,stimulus[i] == first][0:n/3])
        tmp[2].append(data[i,stimulus[i] == second][0:n/3])
        tmp[3].append(data[i,stimulus[i] == third][0:n/3])

    final = dict({'mean':[],'sem':[]})

    for i in tmp.keys():
        tmp2 = np.zeros(n/3)
        tmp3 = np.zeros(n/3)
        for k in xrange(len(tmp2)):
            var = []
            for j in xrange(len(tmp[i])):
                if k < len(tmp[i][j]):
                    var.append(tmp[i][j][k])
            tmp2[k] = np.mean(var)
            tmp3[k] = sem(var)                
        final['mean'].append(tmp2)
        final['sem'].append(tmp3)
        #final['mean'].append(np.mean(tmp[i], 0))
        #final['sem'].append(sem(tmp[i], 0))
    final['mean'] = np.array(final['mean'])
    final['sem'] = np.array(final['sem'])

    if incorrect_trials:
        print "Number of incorrect trials : %.2f" % (incorrect_trials/float(m))+" %"
    if bad_trials:
        print "Number of bad trials : %.2f" % (bad_trials/float(m))+" %"

    return final

def extractStimulusPresentation2(data, stimulus, action, responses):
    """ Return dict
    dict[1] = 0/1 for one-error stimulus
    dict[2] = 0/1 for three-error stimulus
    dict[3] = 0/1 for four-error stimulus
    """
    tmp = dict({1:[],2:[],3:[]})
    m, n = responses.shape
    assert(stimulus.shape == responses.shape == action.shape == data.shape)
    bad_trials = 0
    incorrect_trials = 0

    for i in xrange(m):        
        first, second, third = searchStimOrder(stimulus[i], action[i], responses[i])
        tmp[1].append(data[i,stimulus[i] == first][0:n/3])
        tmp[2].append(data[i,stimulus[i] == second][0:n/3])
        tmp[3].append(data[i,stimulus[i] == third][0:n/3])

    sett = dict()
    for i in tmp.keys():
        sett[i] = dict({j:[] for j in range(12)})
        for j in xrange(len(tmp[i])):
            for k in xrange(len(tmp[i][j])):
                sett[i][k].append(tmp[i][j][k])
        for k in xrange(12):
            sett[i][k] = np.array(sett[i][k])
    return sett
    # m, n = responses.shape
    # assert(stimulus.shape == responses.shape == action.shape == data.shape)
    
    # length = np.zeros((m, 3))
    # order = np.zeros((m, 3))
    # for i in xrange(m):
    #     first, second, third = searchStimOrder(stimulus[i], action[i], responses[i])
    #     order[i, 0] = first
    #     order[i, 1] = second
    #     order[i, 2] = third
    #     length[i, 0] = len(data[i, stimulus[i] == first][0:n/3])
    #     length[i, 1] = len(data[i, stimulus[i] == second][0:n/3])
    #     length[i, 2] = len(data[i, stimulus[i] == third][0:n/3])
        
    # l = np.min(length)
    # tmp = dict()
    # for j in xrange(3):
    #     tmp[j+1] = np.zeros((m, l))
    # for i in xrange(m):
    #     for j in xrange(3):
    #         tmp[j+1][i] = data[i, stimulus[i] == order[i, j]][0:l]

    # return tmp

    
def convertStimulus(state):
    return (state == 's1')*1+(state == 's2')*2 + (state == 's3')*3
def convertAction(action):
    return (action=='thumb')*1+(action=='fore')*2+(action=='midd')*3+(action=='ring')*4+(action=='little')*5


def computeSingleCorrelation(human, model, case = 'JSD'):
	"""Entry should be single-trial vector 
	of performance for each model
	case can be :
	- "JSD" : Jensen-Shannon Divergence
	- "C" : contingency coefficient of Pearson
	- "phi"  phi coefficient
	- "Z" : test Z 
	"""
	h = len(human)
	m = len(model)
	a = float(np.sum(human == 1))
	b = float(np.sum(model == 1))
	obs = np.array([[a, h-a], [b,m-b]])
	h1 = float(a)/float(h)
	m1 = float(b)/float(m)
	h0 = 1-h1
	m0 = 1-m1
	if case == "JSD":
                M1 = np.mean([h1, m1])
                M0 = np.mean([h0, m0])
                dm = computeSpecialKullbackLeibler(np.array([m0, m1]), np.array([M0, M1]))
                dh = computeSpecialKullbackLeibler(np.array([h0, h1]), np.array([M0, M1]))
                return 1-np.mean([dm, dh])

	# elif case == "C":
	#     if h1 == m1:
	# 	return 1
	#     else:
	# 	chi, p, ddl, the = chi2_contingency(obs, correction=False)
	# 	return (chi/(chi+(h+m)))**(0.5)
	# 	#return np.sqrt(chi)

	# elif case == "phi":
	#     if h1 == m1:
        #         return 1
	#     else:
	# 	chi, p, ddl, the = chi2_contingency(obs, correction=False)
	# 	return np.sqrt(chi)

	elif case == "Z":
	    ph1 = float(a)/h
	    pm1 = float(b)/m
	    p = np.mean([ph1, pm1])
	    if ph1 == pm1:
		return 1.0
            elif pm1 == 0.0:
		z = (np.abs(h1-m1))/(np.sqrt(p*(1-p)*((1/a))))
		return 1-(stats.norm.cdf(z, 0, 1)-stats.norm.cdf(-z, 0, 1))                
	    else:
		z = (np.abs(h1-m1))/(np.sqrt(p*(1-p)*((1/a)+(1/b))))
		return 1-(stats.norm.cdf(z, 0, 1)-stats.norm.cdf(-z, 0, 1))

def computeSpecialKullbackLeibler(p, q):
	# Don;t use for compute Divergence Kullback-Leibler
	assert len(p) == len(q)
	tmp = 0
	for i in xrange(len(p)):
	    if q[i] <> 0.0 and p[i] <> 0.0:
		tmp+=p[i]*np.log2(p[i]/q[i])
        return tmp


def computeDistanceMatrix(state, indice):
    m, n = indice.shape    
    distance = np.zeros((m,n))
    for i in xrange(m):        
        for j in xrange(n):
            if indice[i, j] > 1:
                distance[i,j] = j-np.where(state[i,0:j] == state[i,j])[0][-1]

    return distance




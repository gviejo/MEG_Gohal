#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ColorAssociationTask.py

Class that implement the visuo-motor learning task
as described in Brovelli & al, 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *

class CATS():
    """ Class that implement the visuo-motor learning task
    as described in Brovelli & al, 2011 """
    
    def __init__(self, nb_trials = 42):
        self.nb_trials = nb_trials
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.asso = self.createAssociationDict(self.states)
        self.incorrect = np.zeros((len(self.states), len(self.actions)))
        self.stimuli = self.createStimulusList()
        self.order = self.createOrderCorrect(self.states, [1,3,4])
        self.used = []
        self.correct = dict.fromkeys(self.states)
        self.one_error = False
        self.three_error = False
        self.four_error = False
        ## Devaluation ##
        self.devaluation_interval = -1
        self.interval = dict(zip(self.states,np.zeros(len(self.states),dtype='int')))
        ################

    def reinitialize(self):
        #self.__init__(self.nb_trials)
        self.asso = self.createAssociationDict(self.states)
        self.incorrect = np.zeros((len(self.states), len(self.actions)))
        self.stimuli = self.createStimulusList()
        self.order = self.createOrderCorrect(self.states, [1,3,4])
        self.used = []
        self.correct = dict.fromkeys(self.states)
        self.one_error = False
        self.three_error = False
        self.four_error = False
        ## Devaluation ##
        self.devaluation_interval = -1
        self.interval = dict(zip(self.states,np.zeros(len(self.states),dtype='int')))
        ################

    def createOrderCorrect(self, states, nb_incorrect):
        s = list(states)
        np.random.shuffle(s)
        tmp = dict()
        for i,j in zip(s, nb_incorrect):
            tmp[i] = j
        return tmp

    def createAssociationDict(self, states):
        tmp = dict()
        for i in states:
            tmp[i] = dict()
        return tmp

    def getStimulus(self, iteration):
        try:
            if self.interval[self.stimuli[iteration]] == self.devaluation_interval:
                self.devaluate(self.stimuli[iteration])
            return self.stimuli[iteration]
        except:
            print "Error: no more stimuli"
            sys.exit(0)

    def createStimulusList(self):
        s = np.tile(np.array(self.states), ((self.nb_trials/len(self.states))+1, 1))
        map(np.random.shuffle, s)
        return s.flatten()


    def getOutcome(self, state, action, case='meg'):
        if case=='fmri':            
            if state in self.asso.keys() and action in self.asso[state].keys():
                return self.asso[state][action]
            elif state == 's1':                
                if np.sum(self.incorrect[0]==-1)==0:
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1
                elif np.sum(self.incorrect[0]==-1)==1 and 1 not in self.asso[state].values() and not self.one_error:
                    self.one_error = True
                    self.incorrect[self.states.index(state),self.actions.index(action)] = 1
                    self.asso[state][action]=1
                    return 1
                elif 1 in self.asso[state].values():
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1

            elif state == 's2':
                if np.sum(self.incorrect[1]==-1)<3:
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1
                elif np.sum(self.incorrect[1]==-1)==3 and 1 not in self.asso[state].values() and not self.three_error:
                    self.three_error = True
                    self.incorrect[self.states.index(state),self.actions.index(action)] = 1
                    self.asso[state][action]=1
                    return 1
                elif 1 in self.asso[state].values():
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1

            elif state == 's3':
                if np.sum(self.incorrect[2]==-1)<4:
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1
                elif np.sum(self.incorrect[2]==-1)==4 and 1 not in self.asso[state].values() and not self.four_error:
                    self.four_error = True
                    self.incorrect[self.states.index(state),self.actions.index(action)] = 1
                    self.asso[state][action]=1
                    return 1
                elif 1 in self.asso[state].values():
                    self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                    self.asso[state][action]=-1
                    return -1

        elif case=='meg':
            if state in self.asso.keys() and action in self.asso[state].keys():
                if 1 in self.asso[state].values():
                    self.interval[state]+=1
                return self.asso[state][action]
            elif np.sum(self.incorrect[self.states.index(state)] == -1) == 1 and 1 not in self.asso[state].values() and not self.one_error:
                self.one_error = True
                self.asso[state][action] = 1
                self.used.append(action)
                self.correct[state] = action            
                return 1
            elif np.sum(self.incorrect[self.states.index(state)] == -1) == 3 and 1 not in self.asso[state].values() and not self.three_error:
                self.three_error = True
                self.asso[state][action] = 1
                self.used.append(action)
                self.correct[state] = action            
                return 1
            elif np.sum(self.incorrect[self.states.index(state)] == -1) == 4 and 1 not in self.asso[state].values() and not self.four_error:
                self.four_error = True
                self.asso[state][action] = 1
                self.used.append(action)
                self.correct[state] = action            
                return 1
            else:
                self.incorrect[self.states.index(state),self.actions.index(action)] = -1
                self.asso[state][action] = -1
                return -1
###################################################
# devaluation                                     #
###################################################

    def set_devaluation_interval(self, int):
        self.devaluation_interval = int

    def devaluate(self, state):
        rest = list(set(self.actions)-set([self.correct[state]]))
        self.asso[state][self.correct[state]] = -1
        self.correct[state] = np.random.choice(rest)
        self.asso[state][self.correct[state]] = 1



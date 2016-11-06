#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
HumanLearning.py

class to load and analyse data from Brovelli
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
from fonctions import *
# from LearningAnalysis import SSLearning
import scipy.io


class HLearning():
    def __init__(self, directory):
        #directory = dict({'meg':'(PEPS_GoHaL/Beh_Model',42),'fmri':('fMRI/',39)})
        self.directory = directory
        self.nStepEm = 200
        self.pOutset = 0.2
        # -----------------------------------
        # Loading human data
        # -----------------------------------
        self.responses = dict()
        self.stimulus = dict()
        self.action = dict()
        self.reaction = dict()
        self.indice = dict()
        self.subject = dict()
        self.length = dict()
        self.weight = dict()
        for k in self.directory.iterkeys():
            self.responses[k] = []
            self.stimulus[k] = []
            self.action[k] = []
            self.reaction[k] = []
            self.indice[k] = []
            self.weight[k] = []
            self.subject[k] = dict()
            if k is 'meg':
                data = self.loadDirectoryMEG(self.directory[k][0])
            elif k is 'fmri':
                data = self.loadDirectoryfMRI(self.directory[k][0])
            else:
                sys.stderr.write("Unknow directory provided : "+str(self.directory[k]))
                sys.exit(0)            
            self.extractData(data, k, self.directory[k][1])
        self.getExpLength()
        
    def extractData(self, data, case, size):
        for i in data.iterkeys():            
            self.subject[case][i] = dict()
            
            # tmp = np.array([data[i][j]['RT'].flatten()[0:size] for j in data[i].iterkeys()])
            # tmp = tmp-np.mean(tmp)
            # tmp = tmp/np.std(tmp)
            # for j,k in zip(data[i].iterkeys(),tmp):
            #     data[i][j]['RT'] = k
            
            for j in data[i].iterkeys():
                self.responses[case].append(data[i][j]['sar'][0:size,2])
                self.stimulus[case].append(data[i][j]['sar'][0:size,0])
                self.action[case].append(data[i][j]['sar'][0:size,1])
                #self.reaction[case].append(data[i][j]['time'][0:size,1]-data[i][j]['time'][0:size,0])
                if len(data[i][j]['RT'][0]) == 2:
                    self.reaction[case].append(data[i][j]['RT'][:,0][0:size])
                    self.weight[case].append(data[i][j]['RT'][:,1][0:size])
                else:
                    self.reaction[case].append(data[i][j]['RT'].flatten()[0:size])
                #########
                self.subject[case][i][j] = dict({'sar':data[i][j]['sar'],
                                                 'rt':data[i][j]['RT']})                
                #########
        self.responses[case] = np.array(self.responses[case])
        self.stimulus[case] = np.array(self.stimulus[case])
        self.action[case] = np.array(self.action[case])
        self.reaction[case] = np.array(self.reaction[case])
        self.weight[case] = np.array(self.weight[case])

    def loadDirectoryMEG(self, direct):
        data = dict()
        line = "ls "+direct
        p = os.popen(line, "r").read()
        files = p.split('\n')[:-1]
        for i in files:
            if i != 'S1':
                data[i] = dict()
                tmp = scipy.io.loadmat(direct+i+'/beh.mat')['beh']
                for j in range(1, len(tmp[0])-1):
                    data[i][j] = {}
                    for k in range(len(tmp.dtype.names)):
                        data[i][j][tmp.dtype.names[k]] = tmp[0][j][k]   
                tmp = scipy.io.loadmat(direct+i+'/NC.mat')['NC']
                for j in range(1, len(tmp[0])-1):
                    for k in range(len(tmp.dtype.names)):
                        data[i][j][tmp.dtype.names[k]] = tmp[0][j][k]
        return data

    def loadDirectoryfMRI(self, direct):
        data = dict()
        tmp = scipy.io.loadmat(direct+'/beh_allSubj.mat')['data']
        m, n = tmp.shape
        for i in xrange(m):
            sujet = str(tmp[i][0][-1][0]).split("\\")[-2]
            data[sujet] = dict()
            for j in xrange(n):
                num = int(list(str(tmp[i][j][-1][0]).split("\\")[-1])[-1])
                data[sujet][num] = dict()
                for k in range(len(tmp[i][j].dtype.names)):
                    if tmp[i][j].dtype.names[k] == 'sar_time':
                        data[sujet][num]['time'] = tmp[i][j][k]
                    else:
                        data[sujet][num][tmp[i][j].dtype.names[k]] = tmp[i][j][k]                                   
        return data

    #-----------------------------------
    #fitting state space model
    # -----------------------------------
    def stateSpaceAnalysis(case = 'meg'):
        ss = SSLearning(len(self.responses[case][0]), self.pOutset)
        p = []
        for r in self.responses[case]:
            ss.runAnalysis(r, self.nStepEm)
            p.append(ss.pmode)
        p = np.array(p)

        # index of performance
        Ipm = np.log2(p/pOutset)

        # index of cognitive control
        Icc = np.zeros(self.responses[case].shape)
        for i in xrange(Icc.shape[0]):
            j = np.where(self.responses[case][i] == 1)[0][2]
            Icc[i,0:j+1] = -np.log2(1-p[i, 0:j+1])
            Icc[i,j+1:-1] = -np.log2(p[i, j+1:-1])

        # stimulus driven index
        Psd = np.ones(self.responses[case].shape)
        for s in xrange(1,4):
            for i in xrange(Psd.shape[0]):
                for j in xrange(Psd.shape[1]):
                    n = np.sum(self.stimulus[case][i, 0:j]==s)
                    k = np.sum(self.responses[case][i, 0:j][stimulus[i,0:j]==s])
                    Psd[i, j] = Psd[i, j] * binom.cdf(k, n, pOutset)            
        Isd = np.log2(Psd/(0.5**3))
        
        # le papier cadeau
        return dict({'p':p,
                     'Ipm':Ipm,
                     'Icc':Icc,
                     'Psd':Psd,
                     'Isd':Isd})
    

    def computePCA(self, case, step, indice, size = 15):
        tmp = np.zeros((len(indice), size))
        bad = []
        for i in xrange(len(indice)):
            for j in xrange(size):
                if len(self.reaction[case][i][indice[i] == j+1]):
                    tmp[i, j] = np.mean(self.reaction[case][i][indice[i] == j+1])                    
                else:
                    bad.append(i)
        return np.delete(tmp, np.unique(bad), 0)


    def extractRTSteps(self, case, step, indice, size = 15, keep_bad = True):
        tmp = np.zeros((len(indice), size))
        bad = []
        for i in xrange(len(indice)):
            for j in xrange(size):
                if len(self.reaction[case][i][indice[i] == j+1]):
                    tmp[i, j] = np.mean(self.reaction[case][i][indice[i] == j+1])                    
                else:
                    bad.append(i)
        if keep_bad:
            return tmp
        else:            
            return np.delete(tmp, np.unique(bad), 0)        

    def computeIndividualPerformances(self):
        indi = dict()
        for s in self.subject['meg'].iterkeys():
            tmp = 0.0
            size = 0.0
            for i in self.subject['meg'][s].iterkeys():
                tmp+=np.sum(self.subject['meg'][s][i]['sar'][:,2])
                size+=len(self.subject['meg'][s][i]['sar'][:,2])
            indi[s] = tmp/size
        return indi

    def getExpLength(self):
        for k in self.subject.iterkeys():
            self.length[k] = dict()
            for s in self.subject[k].iterkeys():
                tmp = 0
                for t in self.subject[k][s].iterkeys():
                    tmp = tmp+len(self.subject[k][s][t]['rt'])
                self.length[k][s] = float(tmp)


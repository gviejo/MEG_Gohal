#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sferes.py

    class for multi-objective optimization
    to interface with sferes2 : see
    http://sferes2.isir.upmc.fr/
    fitness function is made of Bayesian Information Criterion
    and either Linear Regression
    or possible Reaction Time Likelihood

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import mmap
import numpy as np
if os.uname()[1] in ['paradise']:
    from multiprocessing import Pool, Process
    from pylab import *

#from fonctions import *
from Selection import *
from Models import *
# from HumanLearning import HLearning
#from ColorAssociationTasks import CATS
#from scipy.stats import sem
#from scipy.stats import norm
#from scipy.optimize import leastsq

def unwrap_self_load_data(arg, **kwarg):
    return pareto.loadPooled(*arg, **kwarg)

def unwrap_self_re_test(arg, **kwarg):
    return pareto.poolTest(*arg, **kwarg)

class EA():
    """
    Optimization is made for one subject
    """
    def __init__(self, data, subject, ptr_model):
        self.model = ptr_model
        self.subject = subject
        self.data = data        
        self.n_rs = 15
        self.mean = np.zeros((2,self.n_rs))
        self.fit = np.zeros(2)
        self.rt = self.data['rt'] # array (4*39)
        self.state = self.data['state'] # array int (4*39) 
        self.action = self.data['action'] # array int (4*39)
        self.responses = self.data['reward'] # array int (4*39)
        self.indice = self.data['indice'] # array int (4*39)
        self.mean[0] = self.data['mean'][0] # array (15) centered on median for human
        self.n_trials = self.state.shape[1]
        self.n_blocs = self.state.shape[0]

    def getFitness(self):
        np.seterr(all = 'ignore')        
        for i in xrange(self.n_blocs):
            self.model.startBloc()
            for j in xrange(self.n_trials):            
                self.model.computeValue(self.state[i%self.n_blocs,j]-1, self.action[i%self.n_blocs,j]-1, (i,j))
                self.model.updateValue(self.responses[i%self.n_blocs,j])                                    
        self.fit[0] = float(np.sum(self.model.value))
        self.alignToMedian()        
        self.fit[1] = float(-self.leastSquares())                        
        self.fit[np.isnan(self.fit)] = -1000000.0
        self.fit[np.isinf(self.fit)] = -1000000.0                
        choice = str(self.fit[0]+2000.0)
        rt = str(self.fit[1]+500.0)
        # FUCKING UGLY ########
        if choice == '0.0' or choice == '0': choice = '-100000.0'
        if rt == '0.0' or rt == '0': rt = '-100000.0'
        #######################
        
        return choice, rt

    def leastSquares(self):                
        # for i in xrange(self.n_rs):            
        #     self.mean[1,i] = np.mean(self.model.reaction[self.indice == i+1])
        self.tmp2 = np.zeros(15)
        indice = self.indice.flatten()
        rt = self.model.reaction.flatten()
        for i in xrange(self.n_trials*self.n_blocs):
            if indice[i]-1 < 15:
                self.mean[1][indice[i]-1] += rt[i]
                self.tmp2[indice[i]-1] += 1.0

        error = 0.0
        for i in xrange(15):            
            self.mean[1][i] = self.mean[1][i]/self.tmp2[i]
            error += np.power(self.mean[1][i] - self.mean[0][i], 2)
        return error
        # return np.sum(np.power(self.mean[0]-self.mean[1], 2))            
        #return np.sum(np.power(self.errfunc(p[0], mean[1][0], mean[0][0]), 2))

    def alignToMedian(self):        
        # self.model.reaction = self.model.reaction - np.median(self.model.reaction)
        # self.model.reaction = self.model.reaction / (np.percentile(self.model.reaction, 75)-np.percentile(self.model.reaction, 25))        
        daArray = self.model.reaction.flatten()        
        iSize = len(daArray)
        dpSorted = np.zeros(iSize)
        for i in xrange(iSize):
            dpSorted[i] = daArray[i]
        for i in np.arange(1, 192, 1)[::-1]:
            for j in np.arange(0, i, 1):                
                if dpSorted[j] > dpSorted[j+1]:
                    dTemp = dpSorted[j]
                    dpSorted[j] = dpSorted[j+1]
                    dpSorted[j+1] = dTemp
        dMedian = dpSorted[int((iSize/2)-1)]+(dpSorted[int(iSize/2)]-dpSorted[int((iSize/2)-1)])/2.0;    
        for i in np.arange(0, iSize, 1):
            daArray[i] = daArray[i]-dMedian
            dpSorted[i] = dpSorted[i]-dMedian

        dQ1 = dpSorted[int((iSize/4)-1)]+((dpSorted[int((iSize/4))]-dpSorted[int((iSize/4)-1)])/2.0);
        dQ3 = dpSorted[int((iSize/4)*3-1)]+((dpSorted[int((iSize/4)*3+1)]-dpSorted[((iSize/4)*3-1)])/2.0);
        for i in np.arange(0, iSize, 1):
            daArray[i] = daArray[i]/(dQ3-dQ1)
        self.model.reaction = daArray.reshape(self.n_blocs, self.n_trials)

class RBM():
    """
    Restricted Boltzman machine
    x : Human reaction time
    y : Model Inference
    """
    def __init__(self, x, y, nh = 10, nbiter = 1000):
        # Parameters
        self.nh = nh
        self.nbiter = nbiter
        self.nx = x.shape[1]
        self.ny = y.shape[1]
        self.nd = x.shape[0]
        self.nv = self.nx+self.ny
        self.sig = 0.2        
        self.epsW = 0.5
        self.epsA = 0.5
        self.cost = 0.00001
        self.momentum = 0.95
        # data        
        self.x = np.hstack((x, y))
        self.xx = np.zeros(self.x.shape)  # TEST DATASET
        # Weights
        self.W = np.random.normal(0, 0.1,size=(self.nh+1,self.nv+1))        
        self.dW = np.random.normal(0, 0.001, size = (self.nh+1,self.nv+1))
        # Units
        self.Svis = np.zeros((self.nv+1))                
        self.Svis[-1] = 1.0
        self.Shid = np.zeros((self.nh+1))        
        # Gradient
        self.Wpos = np.zeros((self.nh+1,self.nv+1))
        self.Wneg = np.zeros((self.nh+1,self.nv+1))
        self.apos = np.zeros((self.nd, self.nh+1))
        self.aneg = np.zeros((self.nd, self.nh+1))        
        # Biais
        self.Ahid = np.ones(self.nh+1)
        self.Avis = 0.1*np.ones(self.nv+1)
        self.dA = np.zeros(self.nv+1)
    
        self.Error = np.zeros(self.nbiter)

    def sigmoid(self, x, a):
        return 1.0/(1.0+np.exp(-a*x))

    # visible=0, hidden=1
    def activ(self, who):
        if(who==0):
            self.Svis = np.dot(self.Shid, self.W) + self.sig*np.random.standard_normal(self.nv+1)         
            self.Svis = self.sigmoid(self.Svis, self.Avis)
            self.Svis[-1] = 1.0 # bias
        if(who==1):
            self.Shid = np.dot(self.Svis, self.W.T) + self.sig*np.random.standard_normal(self.nh+1)
            self.Shid = self.sigmoid(self.Shid, self.Ahid)
            #self.Shid = (self.Shid>np.random.rand(self.Shid.shape[0]))*1.0
            self.Shid[-1] = 1.0 # bias        

    def train(self):        
        for i in xrange(self.nbiter):            
            self.Wpos = np.zeros((self.nh+1,self.nv+1))
            self.Wneg = np.zeros((self.nh+1,self.nv+1))
            self.apos = np.zeros((self.nh+1))
            self.aneg = np.zeros((self.nh+1))
            error = 0.0
            for point in xrange(self.nd):
                # Positive Phase
                self.Svis[0:self.nv] = self.x[point]
            
                self.activ(1)            
                self.Wpos = self.Wpos + np.outer(self.Shid, self.Svis)            
                self.apos = self.apos + self.Shid*self.Shid
                # Negative Phase                
                self.activ(0)
                self.activ(1)

                error += np.sum(np.power(self.Svis[0:self.nv]-self.x[point], 2))
                
                # Update phase                
                self.Wneg = self.Wneg + np.outer(self.Shid, self.Svis)
                self.aneg = self.aneg + self.Shid*self.Shid        

            self.Error[i] = error
            self.dW = self.dW*self.momentum + self.epsW * ((self.Wpos - self.Wneg)/float(self.nd) - self.cost*self.W)
            self.W = self.W + self.dW
            self.Ahid = self.Ahid + self.epsA*(self.apos - self.aneg)/(float(self.nd)*self.Ahid*self.Ahid)

            print "Epoch "+str(i)+" Error = "+str(error)

    def getInputfromOutput(self,xx, n = 15):
        self.xx = xx
        self.out = np.zeros((self.xx.shape[0], self.nx))
        for point in xrange(self.xx.shape[0]):
            self.Svis[0:self.nv] = 0.0
            self.Svis[self.nx:self.nx+self.ny] = self.xx[point]
            for i in xrange(n):
                self.activ(1)
                self.activ(0)
                self.Svis[self.nx:self.nx+self.ny] = self.xx[point]
            self.out[point] = self.Svis[0:self.nx]
        return self.out

    def reconstruct(self, xx, n = 10):
        self.xx = xx
        self.out = np.zeros(self.xx.shape)
        for point in xrange(self.xx.shape[0]):
            self.Svis[0:self.nv] = self.xx[point]
            for i in xrange(n):
                self.activ(1)
                self.activ(0)
            self.out[point] = self.Svis[0:self.nv]
        return self.out


class pareto():
    """
    Explore Pareto Front from Sferes Optimization
    """
    def __init__(self, directory, case = 'fmri'):
        self.directory = directory        
        self.case = case
        if case == 'fmri':
            self.N = 156
            self.human = dict({s_dir.split(".")[0]:self.pickling("fmri/"+s_dir) for s_dir in os.listdir("fmri/")})        
        elif case == 'meg':
            self.N = 192
            self.human = dict({s_dir.split(".")[0]:self.pickling("meg/"+s_dir) for s_dir in os.listdir("meg/")})        
        # loading pre-treated data for fmri or meg        
        
        self.data = dict()
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.models = dict({"fusion":FSelection(self.states, self.actions),
                            "qlearning":QLearning(self.states, self.actions),
                            "bayesian":BayesianWorkingMemory(self.states, self.actions),
                            "selection":KSelection(self.states, self.actions),
                            "mixture":CSelection(self.states, self.actions)})

        self.p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma'], 
                            'qlearning':['alpha','beta', 'sigma'],
                            'bayesian':['length','noise','threshold', 'sigma'],
                            'selection':['beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
                            'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma']})

        self.m_order = ['qlearning', 'bayesian', 'selection', 'fusion', 'mixture']
        self.colors_m = dict({'fusion':'r', 'bayesian':'g', 'qlearning':'grey', 'selection':'b', 'mixture':'y'})
        self.opt = dict()
        self.pareto = dict()
        self.distance = dict()
        self.owa = dict()
        self.tche = dict()
        self.p_test = dict()
        self.mixed = dict()
        self.beh = dict({'state':[],'action':[],'responses':[],'reaction':[]})
        self.indd = dict()
        self.zoom = dict()
        self.timing = dict()
        if 'one' in self.directory:
            self.simpleLoadData()
        elif 'two' in self.directory:
            self.loadData()
            # self.simpleLoadData()
        
        

    def showBrute(self):
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8        
        fig_brute = figure(figsize = (10,10)) # for each model all subject            
        axes = {}        
        for s,i in zip(self.human.keys(),range(2,16)):
            axes[s] = fig_brute.add_subplot(4,4,i)

        for s in self.human.iterkeys():            
            for m in self.data.iterkeys():
                if s in self.data[m].keys():
                    tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                    tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                    ind = tmp[:,3] != 0
                    tmp = tmp[ind]
                    tmp = tmp[tmp[:,3].argsort()][::-1]
                    pareto_frontier = [tmp[0]]
                    for pair in tmp[1:]:
                        if pair[4] >= pareto_frontier[-1][4]:
                            pareto_frontier.append(pair)
                    pareto_frontier = np.array(pareto_frontier)
                    pareto_frontier[:,3] -= (2000+float(len(self.p_order[m]))*np.log(156))                    
                    pareto_frontier[:,4] -= 500
                    axes[s].plot(pareto_frontier[:,3], pareto_frontier[:,4], "-o", color = self.colors_m[m], alpha = 1.0)        
                    axes[s].set_title(s)
                    # axes[s].set_ylim(-10,0.0)

    def pickling(self, direc):
        with open(direc, "rb") as f:
            return pickle.load(f)

    def loadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)

        pool = Pool(len(model_in_folders))
        tmp = pool.map(unwrap_self_load_data, zip([self]*len(model_in_folders), model_in_folders))
        
        for d in tmp:
            self.data[d.keys()[0]] = d[d.keys()[0]]

    def simpleLoadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)
        for m in model_in_folders:
            self.data[m] = dict()
            lrun = os.listdir(self.directory+"/"+m)
            order = self.p_order[m.split("_")[0]][0:-1]
            scale = self.models[m.split("_")[0]].bounds

            for r in lrun:                
                s = r.split("_")[3]                
                n = int(r.split("_")[4].split(".")[0])                
                if s in self.data[m].keys():
                    self.data[m][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)
                else :
                    self.data[m][s] = dict()
                    self.data[m][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)                                
                for p in order:
                    self.data[m][s][n][:,order.index(p)+4] = scale[p][0]+self.data[m][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])

    def loadPooled(self, m):         
        data = {m:{}}
        list_file = os.listdir(self.directory+"/"+m)
        order = self.p_order[m]
        scale = self.models[m].bounds
        for r in list_file:
            s = r.split("_")[3]
            n = int(r.split("_")[4].split(".")[0])
            filename = self.directory+"/"+m+"/"+r            
            nb_ind = int(self.tail(filename, 1)[0].split(" ")[1])
            last_gen = np.array(map(lambda x: x[0:-1].split(" "), self.tail(filename, nb_ind+1))).astype('float')
            if s in data[m].keys():
                data[m][s][n] = last_gen
            else:
                data[m][s] = {n:last_gen}
            for p in order:
                data[m][s][n][:,order.index(p)+4] = scale[p][0]+data[m][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])                    
        return data

    def tail(self, filename, n):
        size = os.path.getsize(filename)
        with open(filename, "rb") as f:
            fm = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
            for i in xrange(size-1, -1, -1):
                if fm[i] == '\n':
                    n -= 1
                    if n == -1:
                        break
            return fm[i+1 if i else 0:].splitlines()


    def constructParetoFrontier(self, case = 'r2'):
        best_log = -4*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2))
        worst_log = self.N*np.log(0.2)
        for m in self.data.iterkeys():
            self.pareto[m] = dict()
            for s in self.data[m].iterkeys():
                # print m, s        
                self.pareto[m][s] = dict()   
                tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                ind = tmp[:,3] != 0
                tmp = tmp[ind]
                tmp = tmp[tmp[:,3].argsort()][::-1]
                pareto_frontier = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[4] >= pareto_frontier[-1][4]:
                        pareto_frontier.append(pair)
                self.pareto[m][s] = np.array(pareto_frontier)

                self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - 2000.0
                self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - 500.0
                if case == 'r2':
                    self.pareto[m][s][:,3] = 1.0 - (self.pareto[m][s][:,3]/(self.N*np.log(0.2)))
                elif case == 'log':
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_log)/(best_log-worst_log)
                elif case == 'bic':
                    self.pareto[m][s][:,3] = 2*self.pareto[m][s][:,3] - float(len(self.p_order[m]))*np.log(self.N)
                    best_bic = 2*best_log - float(len(self.p_order[m]))*np.log(self.N)
                    worst_bic = 2*worst_log - float(len(self.p_order[m]))*np.log(self.N)
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_bic)/(best_bic-worst_bic)
                elif case == 'aic':
                    self.pareto[m][s][:,3] = 2*self.pareto[m][s][:,3] - 2.0*float(len(self.p_order[m]))
                    best_aic = 2*best_log - float(len(self.p_order[m]))*2.0
                    worst_aic = 2*worst_log - float(len(self.p_order[m]))*2.0
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_aic)/(best_aic - worst_aic)
                self.pareto[m][s][:,4] = 1.0 - (-self.pareto[m][s][:,4])/(np.power(2*self.human[s]['mean'][0], 2).sum())
                # on enleve les points negatifs
                self.pareto[m][s] = self.pareto[m][s][(self.pareto[m][s][:,3:5]>0).prod(1)==1]


    def constructMixedParetoFrontier(self):
        # subjects = set.intersection(*map(set, [self.pareto[m].keys() for m in self.pareto.keys()]))
        subjects = self.pareto['fusion'].keys()
        for s in subjects:            
            tmp = []            
            for m in self.pareto.iterkeys():
                if s in self.pareto[m].keys():
                    tmp.append(np.hstack((np.ones((len(self.pareto[m][s]),1))*self.m_order.index(m), self.pareto[m][s][:,0:5])))            
            tmp = np.vstack(tmp)            
            tmp = tmp[tmp[:,4].argsort()][::-1]                        
            if len(tmp):
                self.mixed[s] = []
                self.mixed[s] = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[5] >= self.mixed[s][-1][5]:
                        self.mixed[s].append(pair)
                self.mixed[s] = np.array(self.mixed[s])            

    def removeIndivDoublons(self):
        for m in self.pareto.iterkeys():
            for s in self.pareto[m].iterkeys():
                if len(self.pareto[m][s]):
                    # start at column 5; for each parameters columns, find the minimal number of value
                    # then mix all parameters
                    tmp = np.zeros((len(self.pareto[m][s]),len(self.p_order[m])))
                    for i in xrange(len(self.p_order[m])):
                        tmp[:,i][np.unique(self.pareto[m][s][:,i+5], return_index = True)[1]] = 1.0
                    self.pareto[m][s] = self.pareto[m][s][tmp.sum(1)>0]

    def reTest(self):
        for s in self.extremum.keys():        
        # for s in ['S3']:            
            for m in self.extremum[s].keys():
                parameters = self.extremum[s][m]
                # MAking a sferes call to compute a time conversion
                with open(self.case+"/"+s+".pickle", "rb") as f:
                    data = pickle.load(f)
                self.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
                opt = EA(data, s, self.models[m])                                
                for i in xrange(opt.n_blocs):
                    opt.model.startBloc()
                    for j in xrange(opt.n_trials):
                        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
                        opt.model.updateValue(opt.responses[i,j])
                opt.fit[0] = float(np.sum(opt.model.value))
                # print opt.model.value[1,21]
                # if s == 'S8':
                #     print opt.fit[0] +2000.0               
                #     return opt.model.value
                # Check if coherent                 
                print s, m
                print "\t from test :", np.round(opt.fit[0], 2)
                print "\t from sferes :", self.values[s][m]['log']
                
                # for i in xrange(len(opt.model.value.flatten())):
                #     print opt.model.value.flatten()[i]
    
    def rankDistance(self):
        self.p_test['distance'] = dict()        
        self.indd['distance'] = dict()
        for s in self.mixed.iterkeys():
            self.distance[s] = np.zeros((len(self.mixed[s]), 3))
            self.distance[s][:,1] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-np.ones(2), 2),1))
            ind_best_point = np.argmin(self.distance[s][:,1])
            best_point = self.mixed[s][ind_best_point,4:6]
            self.distance[s][:,0] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-best_point,2),1))
            self.distance[s][0:ind_best_point,0] = -1.0*self.distance[s][0:ind_best_point,0]
            self.distance[s][0:ind_best_point,2] = np.arange(-ind_best_point,0)
            self.distance[s][ind_best_point:,2] = np.arange(0, len(self.distance[s])-ind_best_point)
            # Saving best individual                        
            best_ind = self.mixed[s][ind_best_point]
            self.indd['distance'][s] = best_ind
            m = self.m_order[int(best_ind[0])]            
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['distance'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})

    def rankOWA(self):
        self.p_test['owa'] = dict()
        self.indd['owa'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            value = np.sum(np.sort(tmp)*[0.5, 0.5], 1)
            self.owa[s] = value
            ind_best_point = np.argmax(value)
            # Saving best indivudual
            best_ind = self.mixed[s][ind_best_point]
            self.indd['owa'][s] = best_ind
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['owa'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})            

    def rankTchebytchev(self, lambdaa = 0.5, epsilon = 0.001):
        self.p_test['tche'] = dict()
        self.indd['tche'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            ideal = np.max(tmp, 0)
            nadir = np.min(tmp, 0)
            value = lambdaa*((ideal-tmp)/(ideal-nadir))
            value = np.max(value, 1)+epsilon*np.sum(value,1)
            self.tche[s] = value
            ind_best_point = np.argmin(value)
            # Saving best individual
            best_ind = self.mixed[s][ind_best_point]
            self.indd['tche'][s] = best_ind
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['tche'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})                        

    def writeParameters(self, filename):
        with open(filename, 'w') as f:
            for o in self.p_test.keys():
                f.write(o+"\n")
                for s in self.p_test[o].iterkeys():
                    # f.write(m+"\n")                
                    f.write(s+"\n")
                    # for s in subjects:
                    m = self.p_test[o][s].keys()[0]
                    line=m+"\t"+" \t".join([k+"="+str(np.round(self.p_test[o][s][m][k],4)) for k in self.p_order[m]])+"\tloglikelihood = "+str(self.indd[o][s][4])+"\n"      
                    f.write(line)                
                    f.write("\n")

    def preview(self):
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8        
        fig_model = figure(figsize = (10,10)) # for each model all subject            
                
        for m,i in zip(self.pareto.iterkeys(), xrange(len(self.pareto.keys()))):
            ax2 = fig_model.add_subplot(3,2,i+1)
            for s in self.pareto[m].iterkeys():
                # ax2.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], "-o", alpha = 1.0, label = s)        
                ax2.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], "-o", color = self.colors_m[m], alpha = 1.0)        
            ax2.set_title(m)
            ax2.set_xlim(0,1)
            ax2.set_ylim(0,1)
            ax2.legend()        
        ax4 = fig_model.add_subplot(3,2,6)                                            
        for s in self.mixed.keys():
            for m in np.unique(self.mixed[s][:,0]):
                ind = self.mixed[s][:,0] == m
                ax4.plot(self.mixed[s][ind,4], self.mixed[s][ind,5], 'o-', color = self.colors_m[self.m_order[int(m)]])
                ax4.plot(self.zoom[s][np.argmin(self.zoom[s][:,2]),0], self.zoom[s][np.argmin(self.zoom[s][:,2]),1], '*', markersize = 10)
                # ax4.plot(self.zoom[s][np.argmax(self.zoom[s][:,3]),0], self.zoom[s][np.argmax(self.zoom[s][:,3]),1], '^', markersize = 10)
                # ax4.plot(self.zoom[s][np.argmin(self.zoom[s][:,4]),0], self.zoom[s][np.argmin(self.zoom[s][:,4]),1], 'o', markersize = 10)
        ax4.set_xlim(0,1)
        ax4.set_ylim(0,1)


        #  = figure(figsize = (10,6))                 
        # ax7 = fig_evo.add_subplot(1,2,1)
        # ax8 = fig_evo.add_subplot(1,2,2)
        # for m in self.data.iterkeys():
        #     for s in self.data[m].iterkeys():
        #         tmp1 = []
        #         tmp2 = []                
        #         for g in np.unique(self.data[m][s][0][:,0]):
        #             tmp1.append(self.data[m][s][0][self.data[m][s][0][:,0]==g][0,2])
        #             tmp2.append(self.data[m][s][0][self.data[m][s][0][:,0]==g][0,3])                                
        #         ax7.plot(np.unique(self.data[m][s][0][:,0]), np.array(tmp1), 'o-', color = self.colors_m[m])
        #         ax8.plot(np.unique(self.data[m][s][0][:,0]), np.array(tmp2), 'o-', color = self.colors_m[m])
    
        # fig_zoom = figure(figsize = (5,5))
        # ax6 = fig_zoom.add_subplot(1,1,1)
        # for s in self.zoom.keys():            
        #     ax6.plot(self.zoom[s][:,0], self.zoom[s][:,1], '.-', color = 'grey')
        #     ax6.plot(self.zoom[s][np.argmin(self.zoom[s][:,2]),0], self.zoom[s][np.argmin(self.zoom[s][:,2]),1], '*', markersize = 15, color = 'blue', alpha = 0.5)
        #     ax6.plot(self.zoom[s][np.argmax(self.zoom[s][:,3]),0], self.zoom[s][np.argmax(self.zoom[s][:,3]),1], '^', markersize = 15, color = 'red', alpha = 0.5)
        #     ax6.plot(self.zoom[s][np.argmin(self.zoom[s][:,4]),0], self.zoom[s][np.argmin(self.zoom[s][:,4]),1], 'o', markersize = 15, color = 'green', alpha = 0.5)
        # ax6.set_xlim(0,1)
        # ax6.set_ylim(0,1)
        
        # fig_front = figure(figsize = (12,12))
        # m = 'fusion'
        # n = 1
        # for i in xrange(len(self.data[m].keys())):
        #     s = self.data[m].keys()[i]
        #     ax9 = fig_front.add_subplot(4,4,i+1)
        #     color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(self.data[m][s][n][:,0])))))
        #     for g in np.unique(self.data[m][s][n][:,0]):
        #         c = next(color)
        #         ind = self.data[m][s][n][:,0] == g
        #         gen = self.data[m][s][n][:,2:4][ind] - [2000.0,500.0]
        #         ax9.plot(gen[:,0], gen[:,1], 'o', c = c)
        #         ax9.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], '-', linewidth = 3, color='black')
        #     ax9.set_xlim(self.front_bounds[s][0], self.best[0])
        #     ax9.set_ylim(self.front_bounds[s][1], self.best[1])
        #     # ax9.axvline(-4*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2)))
        #     ax9.set_title(s)

    def retrieveRanking(self):
        xmin = 0.0
        ymin = 0.0
        for s in self.mixed.iterkeys():
            self.zoom[s] = np.hstack((self.mixed[s][:,4:6], self.distance[s][:,1:2], np.vstack(self.owa[s]), np.vstack(self.tche[s]), np.vstack(self.mixed[s][:,0])))
        
    def rankIndividualStrategy(self):
        # order is distance, owa , tchenbytchev
        data = {}
        p_test = {}
        self.best_ind_single_strategy = dict()
        for m in ['bayesian', 'qlearning']:
            p_test[m] = dict({'tche':dict(),'owa':dict(),'distance':dict()})
            data[m] = dict()
            subjects = self.pareto[m].keys()
            self.best_ind_single_strategy[m] = dict()
            for s in subjects:
                if len(self.pareto[m][s]):                
                    data[m][s] = np.zeros((self.pareto[m][s].shape[0],5))                
                    # pareto position
                    data[m][s][:,0:2] = self.pareto[m][s][:,3:5]
                    # tchenbytchev ranking
                    lambdaa = 0.5
                    epsilon = 0.001
                    tmp = self.pareto[m][s][:,3:5]
                    ideal = np.max(tmp, 0)                
                    nadir = np.min(tmp, 0)
                    value = lambdaa*((ideal-tmp)/(ideal-nadir))
                    value = np.max(value, 1)+epsilon*np.sum(value,1)
                    data[m][s][:,4] = value
                    ind_best_point = np.argmin(value)            
                    best_ind = self.pareto[m][s][ind_best_point]
                    self.best_ind_single_strategy[m][s] = best_ind
                    p_test[m]['tche'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})                                          
                    # owa ranking
                    data[m][s][:,3] = np.sum(np.sort(tmp)*[0.5, 0.5], 1)                    
                    ind_best_point = np.argmax(data[m][s][:,3])            
                    best_ind = self.pareto[m][s][ind_best_point]
                    p_test[m]['owa'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})
                    # distance ranking
                    data[m][s][:,2] = np.sqrt(np.sum(np.power(tmp-np.ones(2), 2),1))
                    ind_best_point = np.argmin(data[m][s][:,2])
                    best_ind = self.pareto[m][s][ind_best_point]
                    p_test[m]['distance'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})
        return data, p_test
            
    def classifySubject(self):
        models = self.data.keys()                
        subjects = self.human.keys()
        self.values = dict()
        self.extremum = dict()                
        for s in subjects:
            self.extremum[s] = dict()            
            self.values[s] = dict()
            data_best_ind = dict()
            for m in models:
                self.extremum[s][m] = dict()
                self.values[s][m] = dict()
                data = []
                for i in self.data[m][s].iterkeys():
                    #max_gen = np.max(self.data[m][s][i][:,0])
                    #size_max_gen = np.sum(self.data[m][s][i][:,0]==max_gen)
                    tmp = np.hstack((np.ones((len(self.data[m][s][i]),1))*i,self.data[m][s][i]))                    
                    #tmp = np.hstack((np.ones((size_max_gen,1))*i,self.data[m][s][i][-size_max_gen:]))
                    data.append(tmp)
                data = np.vstack(data)
                data[:,3]-=2000.0
                # LOG                
                self.values[s][m]['log'] = np.max(data[:,3])
                best_ind = np.argmax(data[:,3])
                data_best_ind[m] = data[best_ind,5:]
                gen = data[best_ind,1]
                ind = data[best_ind,2]
                # print s, data[best_ind,0], gen, ind
                self.extremum[s][m] = dict(zip(self.p_order[m][0:],data_best_ind[m]))                
                # BIC
                self.values[s][m]['bic'] = 2.0*self.values[s][m]['log'] - float(len(self.p_order[m])-1)*np.log(self.N)
               
        self.best_extremum = dict({'bic':{m:[] for m in models},'log':{m:[] for m in models}})
        self.p_test_extremum = dict({'bic':{},'log':{}})
        for s in self.values.iterkeys():
            for o in ['log', 'bic']:
                best = np.argmax([self.values[s][m][o] for m in models])
                self.best_extremum[o][models[best]].append(s)
                self.p_test_extremum[o][s] = {models[best]:self.extremum[s][models[best]]}

        # #writing parameters because fuck you that's why
        # with open("parameters.txt", 'w') as f:
        #     # for m in models:
        #     for s in subjects:
        #         # f.write(m+"\n")                
        #         f.write(s+"\n")
        #         # for s in subjects:
        #         for m in models:                    
        #             line=m+"\t"+" \t".join([k+"="+str(np.round(self.extremum[s][m][k],4)) for k in self.p_order[m][0:-1]])+"\tloglikelihood = "+str(self.values[s][m]['log'])+"\n"      
        #             f.write(line)                
        #         f.write("\n")


        return 

    def timeConversion(self):
        # for o in self.p_test.iterkeys():
        for o in ['owa']:
            self.timing[o] = dict()
            for s in self.p_test[o].iterkeys():
                self.timing[o][s] = dict()
                m = self.p_test[o][s].keys()[0]
                parameters = self.p_test[o][s][m]
                # MAking a sferes call to compute a time conversion
                with open(self.case+"/"+s+".pickle", "rb") as f:
                    data = pickle.load(f)
                self.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
                opt = EA(data, s, self.models[m])                                
                for i in xrange(opt.n_blocs):
                    opt.model.startBloc()
                    for j in xrange(opt.n_trials):
                        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
                        opt.model.updateValue(opt.responses[i,j])
                opt.fit[0] = float(np.sum(opt.model.value)) + 2000.0
                self.timing[o][s][m] = [np.median(opt.model.reaction)]
                opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
                self.timing[o][s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
                opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
                self.timing[o][s][m] = np.array(self.timing[o][s][m])
                opt.fit[1] = float(-opt.leastSquares()) + 500.0
                # # Check if coherent                 
                ind = self.indd[o][s]
                m2 = self.m_order[int(ind[0])]
                if m != m2:
                    sys.exit()
                real = self.data[m][s][int(ind[1])][ind[3]][2:4]
                
                if np.sum(np.round(real,2)==np.round(opt.fit, 2))!= 2:
                    print o, s, m
                    print "from test :", np.round(opt.fit[0], 2), np.round(opt.fit[1], 2)
                    print "from sferes :", np.round(real[0], 2), np.round(real[1], 2)
                    print "\n"

    def timeConversion_singleStrategy(self, p_test, rank):
        o = 'tche'
        timing = {o:{}}
        for m in ['bayesian','qlearning']:
            timing[o][m] = dict()
            for s in p_test[m][o].iterkeys():
                timing[o][m][s] = dict()
                parameters = p_test[m][o][s][m]
                # MAking a sferes call to compute a time conversion
                with open(self.case+"/"+s+".pickle", "rb") as f:
                    data = pickle.load(f)
                self.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
                opt = EA(data, s, self.models[m])                                
                for i in xrange(opt.n_blocs):
                    opt.model.startBloc()
                    for j in xrange(opt.n_trials):
                        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
                        opt.model.updateValue(opt.responses[i,j])                
                # opt.fit[0] = float(np.sum(opt.model.value))
                opt.fit[0] = 1.0 - (float(np.sum(opt.model.value)))/(self.N*np.log(0.2))
                timing[o][m][s][m] = [np.median(opt.model.reaction)]
                opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
                timing[o][m][s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
                opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
                timing[o][m][s][m] = np.array(timing[o][m][s][m])
                opt.fit[1] = float(-opt.leastSquares())
                #opt.fit[1] = 1.0 - (-opt.fit[1])/(np.power(2*self.human[s]['mean'][0], 2).sum())
                # Check if coherent                                 
                real = self.best_ind_single_strategy[m][s][3:5]
                
                if np.sum(np.round(real,2)==np.round(opt.fit, 2))!= 2:
                    print o, s, m
                    print "from test :", opt.fit
                    print "from sferes :", real
        return timing

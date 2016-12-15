#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import cPickle as pickle
import sys, os
import matplotlib
matplotlib.use('Agg')
from mplothelper import MPlotHelper
import matplotlib.pyplot as plt
import os

with open("../pareto2.pickle", 'rb') as handle:
	data = pickle.load(handle)
with open("../pareto3.pickle", 'rb') as handle:
	mixed_front = pickle.load(handle)    
with open("../pareto4.pickle", 'rb') as handle:
	first_front = pickle.load(handle)
with open("../position.pickle", 'rb') as handle:
	pos = pickle.load(handle)
with open("../position_pre_test.pickle", 'rb') as handle:
	pos2 = pickle.load(handle)
with open("../p_test_last_set.pickle", 'rb') as handle:
	p_test = pickle.load(handle)
tmp = {}
tmp2 = {}
for k in pos.iterkeys():
	tmp[k[0:-1]] = pos[k]
for k in pos2.iterkeys():
	tmp2[k[0:-1]] = pos2[k]
pos = tmp
pos2 = tmp2

##################################
# PLOT PARAMETERS ################
##################################
dpi = 900
alpha = 0.8
msize = 1.7
lwidth = 1.0
elwidth = 0.4
cpsize = 1.1
# figure_size = (2049,1600) # In pixels
# subplot_size = (390,280)
# subplot_interspace = (100, 80)
# subplot_margin = (100,100)
# figure_size = (6378,9000) # In pixels
figure_size = (9000,6700) # In pixels
subplot_size = (1600,1700)
subplot_interspace = (380, 320)
subplot_margin = (500,300)
dashes = ['-', '--', ':']
colors = ['blue','red','green']
plt.rc('legend', fontsize='medium', handlelength=1, numpoints=1, frameon=False)
plt.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif', 'size' : 9})
plt.rc('axes', linewidth=1.0)
plt.rc('xtick', direction='out', labelsize='small')
plt.rc('xtick.major', width=1.0)
plt.rc('ytick', direction='out', labelsize='small')
plt.rc('ytick.major', width=1.0)
plt.rc('text',usetex=True)
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
colors_m = dict({'fusion':'#F1433F',
				'bayesian':'#D5A253',
				'qlearning': '#6E8243',
				'selection':'#70B7BA',
				'mixture':'#3D4C53'})
legend_m = dict({'fusion':r'$Coordination\ par\ Entropie$',
				'bayesian':r'$M\acute{e}moire\ de\ travail\ bay\acute{e}sienne$',
				'qlearning':r'$Q-Learning$',
				'selection': r'$S\acute{e}lection\ par\ VPI$',
				'mixture': r'$M\acute{e}lange\ pond\acute{e}r\acute{e}$'})

positions = []
for i in np.arange(0,1,0.34)[::-1]:
	for j in np.arange(0,1,0.25):
		positions.append([j,i])

positions = np.array(positions)
positions[:,0] = np.floor(figure_size[0]*positions[:,0]+subplot_interspace[0])
positions[:,1] = np.floor(figure_size[1]*positions[:,1]+subplot_interspace[1])

p = MPlotHelper()
p.figure(figure_size[0], figure_size[1], dpi)


markers = ['^', 'o', 'p', 's', 'D', 'H', '|']


xlimit = dict({0:(0.45,0.65),
			   1:(0.45,0.65),
			   2:(0.4,0.6),
			   3:(0.6,0.70),
			   4:(0.6,0.75)})
ylimit = dict({0:(0.4,0.7),
			   1:(0.2,0.7),
			   2:(0.2,0.5),
			   3:(0.64,0.70),
			   4:(0.95,1.0)})

n_subjects = len(data.keys())
subjects= data.keys()

for i in xrange(n_subjects):
	# print subjects[i]
	ax1 = p.subfigure(positions[i], subplot_size)
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	s = subjects[i]

	# # # TOUT LES FRONTS DE CHAQUE MODEL 
	# for m in data[s].keys():	
	# 	positif = (data[s][m][:,4]>0)*(data[s][m][:,5]>0)		
	# 	# les sets dispo dans le front de pareto
	# 	sets = np.unique(data[s][m][positif,0])
	# 	ax1.plot(data[s][m][positif,4], data[s][m][positif,5], '-', color = colors_m[m], linewidth = 1.0)
	# 	for j in sets:
	# 		index = (data[s][m][:,0] == j)*positif
	# 		# ax1.plot(data[s][m][index,4],data[s][m][index,5], markers[int(j-1)], markerfacecolor = 'white',  markeredgewidth = 1.0, markersize = 6.5, markeredgecolor = colors_m[m])
	# 		ax1.plot(data[s][m][index,4], data[s][m][index,5] , markers[int(j-1)], markerfacecolor = 'white',  markeredgewidth = 1.0, markersize = 6.5, markeredgecolor = colors_m[m])


	# LE FRONT MIXE
	positif = (mixed_front[s][:,5]>0)*(mixed_front[s][:,6]>0)		
	ax1.plot(mixed_front[s][positif,5], mixed_front[s][positif,6], '-', color = 'black', linewidth = 1.0)
	models = np.unique(mixed_front[s][:,0])	
	sets = np.unique(mixed_front[s][:,1])		
	for k in models:		 					 
		m = id_to_models[k]
		for j in sets:
			index = (mixed_front[s][:,0] == k)*(mixed_front[s][:,1] == j)*positif
			if index.sum():
				ax1.plot(mixed_front[s][index,5], mixed_front[s][index,6], markers[int(j-1)], markerfacecolor = 'white',  markeredgewidth = 1.0, markersize = 3.5, markeredgecolor = colors_m[m])
	
	# LE PREMIER FRONT
	positif = (first_front[s][:,4]>0)*(first_front[s][:,5]>0)		
	ax1.plot(first_front[s][positif,4], first_front[s][positif,5], '-', color = 'gray', linewidth = 1.0)
	models = np.unique(first_front[s][:,0])		
	for k in models:		 					 
		m = id_to_models[k]	 	
		index = (first_front[s][:,0] == k)*positif
		if index.sum():
			ax1.plot(first_front[s][index,4], first_front[s][index,5], markers[0], markerfacecolor = 'white',  markeredgewidth = 0.5, markersize = 2.5, markeredgecolor = colors_m[m])
				


	ax1.plot(pos[s][0], pos[s][1], '*', markersize = 10, color = 'black')		
	ax1.plot(pos2[s][0], pos2[s][1], 'o', markersize = 10, color = 'black')		

	ax1.locator_params(nbins=5)	


	# ax1.set_xlim(0,1)
	# ax1.set_ylim(0,1)

	ax1.xaxis.labelpad = -2
	
	ax1.set_xlabel("$Choix$", fontsize = 8, labelpad = 2)
	ax1.set_ylabel(r"$Temps\ de\ r\acute{e}action$", fontsize = 8, labelpad = 2)

	# make title with set name
	s = subjects[i]
	set_ = 0
	for t in p_test.keys():
		if s == t[0:-1]:
			set_ = t[-1]
			break
	ax1.set_title(subjects[i]+" version "+set_, y = 0.98)


line2 = tuple([plt.Line2D(range(1),range(1),alpha=1.0,color=colors_m[m], linewidth = 2) for m in colors_m.keys()])
plt.figlegend(line2,tuple(legend_m.values()), loc = 'lower right', bbox_to_anchor = (0.999, 0.05))
line3 = tuple([plt.Line2D(range(1),range(1), linestyle = '', marker = markers[i], alpha=1.0, markerfacecolor = 'white', color='black') for i in xrange(len(markers))])
plt.figlegend(line3,tuple(["Version "+str(i+1) for i in xrange(6)]), loc = 'lower right', bbox_to_anchor = (0.86, 0.2))



plt.savefig('figure_set_meg_pareto.pdf')
# plt.savefig('../../../Dropbox/Soutenance/Slides/figures/figure_set_meg_pareto.pdf')


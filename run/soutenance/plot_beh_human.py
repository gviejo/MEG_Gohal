#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
import cPickle as pickle
import os, sys
import matplotlib
matplotlib.use('Agg')
from mplothelper import MPlotHelper
import matplotlib.pyplot as plt
sys.path.append("../../src")
from fonctions import *

def pickling(direc):
        with open(direc, "rb") as f:
            return pickle.load(f)
human = dict({s_dir.split(".")[0]:pickling("../meg/"+s_dir) for s_dir in os.listdir("../meg/")})        

data = {}
data['state'] = []
data['action'] = []
data['reward'] = []
data['rt'] = []
for s in human.iterkeys():
    if s != 'S1':
        data['state'].append(human[s]['state'])
        data['action'].append(human[s]['action'])
        data['reward'].append(human[s]['reward'])
        data['rt'].append(human[s]['rt'])

for k in data.keys():
    data[k] = np.array(data[k]).reshape(11*4,48)


hpcr = extractStimulusPresentation(data['reward'], data['state'], data['action'], data['reward'])
step, indice = getRepresentativeSteps(data['rt'], data['state'], data['action'], data['reward'], 'meg')
tmp = computeMeanRepresentativeSteps(step) 
hrt = {'mean':tmp[0], 'sem':tmp[1]}




##################################
# PLOT PARAMETERS ################
##################################
dpi = 900.0
alpha = 1.0
size_m = 1.7
lwidth = 1.0
elwidth = 0.4
cpsize = 1.1
# figure_size = (980,1300) # In pixels
# subplot_interspace = (160, 150)
# subplot_size = (780,505)
# subplot_margin = (170,130)
# figure_size = (980,1250) # In pixels for 300 dpi
# subplot_interspace = (160, 150)
# subplot_size = (780,440)
# subplot_margin = (170,130)
# figure_size = (6012,1900) # In pixels for 900 dpi horizontal
figure_size = (3100,4002) # vertical
subplot_interspace = (650, 681)
subplot_size = (2397,1352)
subplot_margin = (522,399)
dashes = ['-', '--', ':']
colors = ['blue','red','green']
plt.rc('legend', fontsize='medium', handlelength=1, numpoints=1, frameon=False)
plt.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif', 'size' : 8})
plt.rc('axes', linewidth=1.0)
plt.rc('xtick', direction='out', labelsize='medium')
plt.rc('xtick.major', width=1.0)
plt.rc('ytick', direction='out', labelsize='medium')
plt.rc('ytick.major', width=1.0)
plt.rc('text',usetex=True)

positions = (subplot_margin, (subplot_margin[0], subplot_margin[1]+subplot_size[1]+subplot_interspace[1]))
# positions = ((subplot_margin[0]+subplot_size[0]+subplot_interspace[0],subplot_margin[1]), subplot_margin)
#####################################
#####################################

p = MPlotHelper()
p.figure(figure_size[0], figure_size[1], dpi)
ax1 = p.subfigure(positions[1], subplot_size)

line1 = tuple([plt.Line2D(range(1),range(1),marker='o',alpha=1.0,color=colors[i], markersize = size_m) for i in xrange(3)])
plt.figlegend(line1,tuple(["S1 : une erreur", "S3 : trois erreurs", "S4 : quatre erreurs"]), loc = 'lower right', bbox_to_anchor = (0.96, 0.65))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
for i in xrange(3):
    ax1.errorbar(range(1, len(hpcr['mean'][i])+1), hpcr['mean'][i], hpcr['sem'][i], alpha = 1.0,
                                                                                    marker = 'o', 
                                                                                    markersize = size_m, 
                                                                                    linestyle = dashes[1], 
                                                                                    color = colors[i], 
                                                                                    linewidth = lwidth, 
                                                                                    elinewidth = elwidth, 
                                                                                    markeredgecolor = colors[i], 
                                                                                    capsize = cpsize)    
    

ax1.set_ylabel("$Performances$")
ax1.set_xlabel("$Essai$")
ax1.locator_params(nbins=3)    
ax1.set_xticks(range(1,18,4))
ax1.set_xlim(0,17)
ax1.set_ylim(0.0, 1.05)



ax2 = p.subfigure(positions[0], subplot_size)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.errorbar(range(1, len(hrt['mean'])+1), hrt['mean'], hrt['sem'], linestyle = '--', alpha = alpha, color = 'grey', 
                                        markeredgecolor='grey',
                                        marker = 'o',
                                        linewidth = lwidth, 
                                        elinewidth = elwidth, 
                                        markersize = 3.5, 
                                        capsize = 1.1)


ax2.set_xlabel(r'$Essai\ repr\acute{e}sentatif$')
ax2.set_ylabel(r'$RT centr\acute{e}$')
ax2.locator_params(nbins=5)
# ###
msize = 3.5
mwidth = 0.5
ax2.plot(1, 0.452, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
ax2.plot(1, 0.442, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(1, 0.432, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(2, 0.452, 'o', color = 'blue', markersize=msize)
ax2.plot(2, 0.442, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(2, 0.432, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(3, 0.442, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax2.plot(3, 0.432, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(4, 0.442, 'o', color = 'red', markersize=msize)
ax2.plot(4, 0.432, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax2.plot(5, 0.432, 'o', color = 'green', markersize=msize)
for i in xrange(6,16,1):
    ax2.plot(i, 0.452, 'o', color = 'blue', markersize=msize)
    ax2.plot(i, 0.442, 'o', color = 'red', markersize=msize)
    ax2.plot(i, 0.432, 'o', color = 'green', markersize=msize)
ax2.set_ylim(0.425, 0.56)
ax2.set_xlim(0,16)
# ax2.set_xticks(range(1,17,2))
# ###

# p.fig.text(0.01,0.93, "A", fontsize = 13)
# p.fig.text(0.01,0.46, "B", fontsize = 13)


plt.savefig('fig_beh_human_meg.pdf')

plt.savefig('../../../Dropbox/Soutenance/Slides/figures/fig_beh_human_meg.pdf')


# plt.savefig('../figures/Fig1.pdf')
# os.system("evince ../figures/Fig1.pdf")


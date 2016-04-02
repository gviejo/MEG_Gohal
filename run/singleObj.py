#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and and plot single objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../src")
from fonctions import *

from Models import *
#from matplotlib import *
#from pylab import *

from Sferes import pareto
from itertools import *

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args()
# -----------------------------------

# -----------------------------------
# LOADING DATA
# -----------------------------------
front = pareto(options.input, case = 'meg')

front.classifySubject()

front.reTest()




#fit to choice extremum of the front
with open("extremum.pickle", 'wb') as f:
	pickle.dump(front.p_test_extremum, f)

# # value of maximum BIC normalized 
# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/obj_choice.pickle"), 'wb') as f:
#   	pickle.dump(dict({'best':front.best_extremum,'values':front.values}), f)

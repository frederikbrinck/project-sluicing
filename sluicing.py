#!/usr/bin/python
import os
import random
import importlib

import kenlm
import numpy as np, numpy.random

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from lib.data import loadData, saveData, splitData, tableFromData, predictData
from lib.functions import getAntecedents, kfoldValidation



# define a context manager to surpress 
# function printing, taken from
# http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class surpressPrint(object):
    def __init__(self):
        # open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



####### ---->
###			------------>
###		LET'S GO 		----------->
###			------------>
####### ---->
if __name__ == '__main__':
	import argparse
    
    # setup parser and parse args
	parser = argparse.ArgumentParser(description='Extracts antecedent data and examples from the provided jsons file.')
	parser.add_argument('dataref', metavar='dataref', type=str, help='Reference to the data file')
	args = parser.parse_args()

	with surpressPrint():
		modelFeatures = []
		lmModel = kenlm.Model('models/test.arpa')
		modelFeatures.append({ "active": 1, "feature":"f_language", "args": [lmModel, 9] })
		modelFeatures.append({ "active": 1, "feature":"f_score", "args": [] })
		modelFeatures.append({ "active": 0, "feature":"f_pos", "args": ["models/table"] })

    # load data from all active features
	examples = loadData(args.dataref)

	totalX = np.zeros((len(examples),0))
	for model in modelFeatures:
		if model["active"]:
			print "Loading features from", model["feature"]

			# load feature data and print sample
			path = "features." + model["feature"]
			feature = importlib.import_module(path)
			try:
				maxFeatures = feature.featureNumber()
				dataX, dataY = feature.extractFeatures(examples, *model["args"])
			except Exception as error:
				print "Error: Feature", model["feature"], "must contain function featureNumber() and extractFeatures(...).", error

			totalX = np.append(totalX, dataX, axis=1)

	# scaling data
	print "Scaling data..."
	totalX = preprocessing.maxabs_scale(totalX)

	# run kfold validation
	print "----------------------------------"
	print "Running kfold validation on all features"
	print "----------------------------------"
	kfoldValidation(10, np.array(totalX), np.array(dataY), True)
#!/usr/bin/python
import os
import sys
import random
import importlib
import itertools
import math

import kenlm
import numpy as np, numpy.random

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from lib.data import loadData, saveData, splitData, filterData, tableFromData, predictData
from lib.functions import getAntecedents, getLengthCounts, kfoldValidation



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



# combine all features given in a 
# feature model into one feature list
def combineFeatures(examples, modelFeatures, verbose=False):
	totalX = np.zeros((len(examples),0))
	for model in modelFeatures:
		if model["active"]:
			if verbose:
				print "Loading features from", model["feature"]

			# load feature data and print sample
			path = "features." + model["feature"]
			feature = importlib.import_module(path)
			try:
				maxCoefs = feature.coefNumber()
				dataX, dataY = feature.extractFeatures(examples, *model["args"], **model["kwargs"])
			except Exception as error:
				print "Error: Feature", model["feature"], "must contain function coefNumber() and extractFeatures(...).", error
				sys.exit(1)

			totalX = np.append(totalX, dataX, axis=1)

	return totalX, dataY



# return the total number of 
# features coefficient for the given
# features of the model
def coefNumber(modelFeatures):
	numCoefs = 0
	for model in modelFeatures:
		if model["active"]:
			# load feature data and print sample
			path = "features." + model["feature"]
			feature = importlib.import_module(path)
			numCoefs += feature.coefNumber()
			

	return numCoefs



####### ---->
###			------------>
###		LET'S GO 		----------->
###			------------>
####### ---->
if __name__ == '__main__':
	import argparse
    
    # setup parser and parse args
	parser = argparse.ArgumentParser(description='Extracts antecedent data and examples from the provided jsons file.')
	parser.add_argument('dataref', metavar='path', type=str, help='Reference to the data file')
	parser.add_argument('--permute', metavar='true/false', type=lambda x: x.lower() in ("yes", "true", "1"), help='Whether or not to do permutations of candidates in each set')
	args = parser.parse_args()

	# add features for svm model
	with surpressPrint():
		modelFeatures = []
		lmModel = kenlm.Model('models/test.arpa')
		modelFeatures.append({ "active": 1, "feature": "f_language", "args": [lmModel, 9], "kwargs": { "prepend": False } })
		modelFeatures.append({ "active": 1, "feature": "f_score", "args": [], "kwargs": { "prepend": False } })
		modelFeatures.append({ "active": 0, "feature": "f_pos", "args": [], "kwargs": { "table": "models/table" } })

    # load data from all active features
	examples = loadData(args.dataref)

	# print counts of candidate sets
	# grouped by length
	if False:
		counts = getLengthCounts(examples)
		print counts
		sys.exit(0)

	# initialise permutations if requested
	if args.permute:
		# filter data and get permutation generators
		length = 4
		data = filterData(examples, length=length)
		for k in data.keys():
			data[k] = itertools.permutations(data[k])

		# recombine data into correct
		totalX = np.zeros((0, length * coefNumber(modelFeatures)))
		totalY = np.zeros(0)
		for i in range(math.factorial(length)):
			examples = {}
			for k in data.keys():
				examples[k] = data[k].next()

			# combine all features for each candidate set
			dataX, dataY = combineFeatures(examples, modelFeatures)
			totalX = np.append(totalX, np.array(dataX), axis=0)
			totalY = np.append(totalY, np.array(dataY))

		totalX = preprocessing.maxabs_scale(totalX)

		# do kfold validation and print result
		accuracies = kfoldValidation(10, np.array(totalX), np.array(totalY), verbose=True)

	# otherwise run standard kfold validation on
	# all of the added features
	else:
		totalX, dataY = combineFeatures(examples, modelFeatures, verbose=True)

		# scaling data
		print "Scaling data..."
		totalX = preprocessing.maxabs_scale(totalX)

		# run kfold validation
		print "----------------------------------"
		print "Running kfold validation on all features"
		print "----------------------------------"
		kfoldValidation(10, np.array(totalX), np.array(dataY), verbose=True)
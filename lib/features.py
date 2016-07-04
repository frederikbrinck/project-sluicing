# features.py contains standard functions
# used for doing feature manipulation
# -----------------------------------------
import importlib
import itertools
import math

import numpy as np
from data import filterData



# return the total number of 
# feature coefficients for the
# features of the given model
def coefNumber(modelFeatures):
	numCoefs = 0
	for model in modelFeatures:
		if model["active"]:
			# load feature data and print sample
			path = "features." + model["feature"]
			feature = importlib.import_module(path)
			numCoefs += feature.coefNumber()
			
	return numCoefs



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
				dataX, totalY = feature.extractFeatures(examples, *model["args"], **model["kwargs"])
			except Exception as error:
				print "Error: Feature", model["feature"], "must contain function coefNumber() and extractFeatures(...).", error
				sys.exit(1)

			totalX = np.append(totalX, dataX, axis=1)

	return totalX, totalY



# create a list of features comprised
# of all features of all candidate sets
# in all their permutations
def permuteFeatures(examples, modelFeatures, length=4):
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

	return totalX, totalY
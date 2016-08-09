#!/usr/bin/python
import os
import sys
import random

import kenlm
import numpy as np, numpy.random

from sklearn import preprocessing
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from lib.data import loadData, saveData, splitData, filterData, tableFromData, predictData
from lib.functions import getAntecedents, getLengthCounts, kfoldValidation
from lib.features import combineFeatures, permuteFeatures



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
	parser.add_argument('dataref', metavar='path', type=str, help='Reference to the data file')
	parser.add_argument('--permute', metavar='permutation length', type=int, help='Whether or not to do permutations of candidates in each set')
	args = parser.parse_args()

	# add features and the classifier
	# for the svm model
	with surpressPrint():
		modelFeatures = []

		# add language model
		lmModel = kenlm.Model('models/standard.arpa')
		modelFeatures.append({ "active": 1, "feature": "f_language", "args": [lmModel, 9], "kwargs": { "prepend": False } })
		
		# add our features, note that the 
		# total amount of features are
		# "sluiceType,distanceFromSluice,sluiceCandidateOverlap,backwards,WH_gov_npmi,containsSluice,isDominatedBySluice,isInRelClause,isInParenthetical,coordWithSluice,immedAfterCataphoricSluice,afterInitialSluice,sluiceInCataphoricPattern,LocativeCorr,EntityCorr,TemporalCorr,DegreeCorr,WhichCorr"
		modelFeatures.append({ "active": 1, "feature": "f_score", "args": [], "kwargs": { "prepend": False, "features": "sluiceType,distanceFromSluice,sluiceCandidateOverlap,backwards,WH_gov_npmi,containsSluice,isDominatedBySluice,isInRelClause,isInParenthetical,coordWithSluice,immedAfterCataphoricSluice,afterInitialSluice,sluiceInCataphoricPattern,LocativeCorr,EntityCorr,TemporalCorr,DegreeCorr,WhichCorr" } })

		# add pos tagging feature
		modelFeatures.append({ "active": 1, "feature": "f_pos", "args": [], "kwargs": { "table": "models/table" } })
		
		clf = OneVsRestClassifier(svm.LinearSVC(random_state=0)) #svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    # load data from all active features
	examples = loadData(args.dataref)

	# print counts of candidate sets
	# grouped by length
	if False:
		counts = getLengthCounts(examples)
		print counts
		sys.exit(0)

	# get permuted features if requested
	if args.permute:
		totalX, totalY = permuteFeatures(examples, modelFeatures, length=args.permute)
	# otherwise get features in a regular format
	else:
		totalX, totalY = combineFeatures(examples, modelFeatures, verbose=True)

	# scaling data
	print "Scaling data..."
	totalX = preprocessing.maxabs_scale(totalX)

	# run kfold validation 
	print "----------------------------------"
	print "Running kfold validation on all features"
	print "----------------------------------"
	kfoldValidation(10, np.array(totalX), np.array(totalY), classifier=clf, verbose=True)

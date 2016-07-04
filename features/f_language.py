# feature script that models
# antecedent detection through
# a language model, picking
# the most likely phrase
# when the antecedent is juxtaposed
# with the sluice

import os
import kenlm
import sys
# import parent libraries
sys.path.insert(0, '.')

from nltk import pos_tag

from lib.data import loadData
from lib.functions import addPadding, kfoldValidation



# return the number of features
# added for this script per candidate
def coefNumber():
	return 1



# use the data table to figure out the 
# probabilities for all data
def extractFeatures(examples, model, n, prepend=False):

	# run over all examples counting
	# the maximum length to allow
	# for zero-padding later on
	dataProbabilities = []
	dataY = []
	maxChunk = 0
	for sluiceId in examples:

		# prepare sentence data
		sentenceProbabilities = []
		chunk = 0
		for sentence in examples[sluiceId]:

			# form sentence to pitch against
			# the language model 
			candidate = sentence["text"]
			sluice = sentence["sluiceGovVPText"]
			pitch = sluice + " " + candidate;
			pitch = pitch.split(" ");
			pitch = " ".join(pitch[0:min(n, len(pitch))])

			# calculate the probability for the 
			# pitch sentence usin the KenLM.
			# Consider switiching to a better
			# language model.
			sentenceProbabilities.append(model.score(pitch))

			# append data if antecedent
			# and update iterator
			if sentence["isAntecedent"]:
				dataY.append(chunk)

			chunk += 1

		# add data for X and
		# update max chunk
		dataProbabilities.append(sentenceProbabilities)
		if chunk > maxChunk:
			maxChunk = chunk

	# before returning, add padding to
	# the data in case some examples
	# have too few sentences
	# for example in dataProbabilities:
	# 	 if len(example) < maxChunk:
	# 	 	for i in range(maxChunk - len(example)):
	# 	 		example.append(0.0)
	dataProbabilities = addPadding(dataProbabilities, maxChunk, prepend)

	return dataProbabilities, dataY



####### ---->
###			------------>
###		LET'S GO 		----------->
###			------------>
####### ---->
if __name__ == '__main__':
	import argparse

	# setup parser and parse args
	parser = argparse.ArgumentParser(description='Trains the parameters of the POS model for antecedent identificaton')
	parser.add_argument('dataref', metavar='dataref', type=str, help='Reference to the example file')
	args = parser.parse_args()
	
	# load language model, and data
	model = kenlm.Model('models/test.arpa')
	examples = loadData(args.dataref)
	
	# run 10-fold cross validation
	# for each type of sentence length
	# in the range and calculate the accuracy
	kfold = 10
	overall = []
	for k in range(5,21):
		dataX, dataY = extractFeatures(examples, model, k)
		accuracies = kfoldValidation(kfold, dataX, dataY, True)
		overall.append(sum(accuracies) / len(accuracies))
	
	print "-------"
	print "Average score:", sum(overall) / len(overall)
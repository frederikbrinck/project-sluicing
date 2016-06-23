# feature script that models
# antecedent detection through 
# ngram probabilities in a
# BLEU-like

import sys
from nltk import pos_tag

from lib.data import loadData
from lib.functions import getSluice, kfoldValidation
from lib.probability import computeProbability



# use the data table to figure out the 
# probabilities for all data
def extractProbabilities(examples, table):
	# load and format data correctly
	probabilities = loadData(table, "key")
	leastNgram = 1
	highestNgram = 3
	for i in range(leastNgram, highestNgram + 1):
		try:
			probabilities[str(i)] = probabilities[str(i)][0]
		except:
			print "Ngrams are not set correctly"
			sys.exit()

	# run over all examples counting
	# the maximum length to allow
	# for zero-padding later on
	dataProbabilities = []
	dataY = []
	maxLength = 0
	for sluiceId in examples:

		# prepare sentence data
		sentenceProbabilities = []
		length = 0
		for sentence in examples[sluiceId]:

			# extract sluice and pos tags
			# from candidate
			candidate = sentence["text"]
			tags = [m[1] for m in pos_tag(candidate)]
			sluice = getSluice(sentence["sluiceGovVPText"])
			if not sluice:
				sluice = sentence["sluiceGovVPText"]

			# calculate probabilities
			for i in range(leastNgram, highestNgram + 1):
				sentenceProbabilities.append(computeProbability(i, tags, sluice, probabilities))

			# append data if antecedent
			# and update iterator
			if sentence["isAntecedent"]:
				dataY.append(length)

			length += 1

		# add data for X and
		# update max length
		dataProbabilities.append(sentenceProbabilities)
		if length > maxLength:
			maxLength = length

	# before returning, add padding to
	# the data in case some examples
	# have too few sentences
	for example in dataProbabilities:
		 if len(example) < (highestNgram - leastNgram + 1) * maxLength:
		 	for i in range((highestNgram - leastNgram + 1) * maxLength - len(example)):
		 		example.append(0.0)
	
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
	parser.add_argument('datatable', metavar='datatable', type=str, help='Reference to the table file')
	args = parser.parse_args()

	# get data and batch size
	kfold = 10
	examples = loadData(args.dataref)

	# get the data in the right format, and
	# run a kfold validation
	dataX, dataY = extractProbabilities(examples, args.datatable)
	kfoldValidation(kfold, dataX, dataY, True)

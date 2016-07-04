# feature script that models
# antecedent detection through 
# ngram probabilities in a
# BLEU-like

import sys
# import parent libraries
sys.path.insert(0, '.')

from nltk import pos_tag

from lib.data import loadData, saveData, tableFromData
from lib.functions import getSluice, kfoldValidation
from lib.probability import computeProbability



# globals
leastNgram = 1
highestNgram = 5



# return the number of features
# added for this script per candidate
def coefNumber():
	global leastNgram, highestNgram
	return highestNgram + 1 - leastNgram



# use the data table to figure out the 
# probabilities for all data
def extractFeatures(examples, table=False):
	global leastNgram, highestNgram

	# load and format data correctly if
	# given a table; otherwise calculate
	# probabilities
	if table:
		probabilities = loadData(table, "key")
		for i in range(leastNgram, highestNgram + 1):
			try:
				probabilities[str(i)] = probabilities[str(i)][0]
			except:
				print "Ngrams are not set correctly"
				sys.exit()
	else:
		probabilities = tableFromData(examples, leastNgram, highestNgram)

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
	
	if table:
		return dataProbabilities, dataY
	else:
		return dataProbabilities, dataY, probabilities



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
	parser.add_argument('-s', '--save', metavar='save', type=str, help='Save the table model to the given distination generated in this pass')
	parser.add_argument('-m', '--model', metavar='model', type=str, default=False, help='Reference to the table file')
	args = parser.parse_args()

	# load data and set 
	# batch size
	examples = loadData(args.dataref)
	kfold = 10

	# get the data in the right format, and
	# run a kfold validation
	if args.model:
		dataX, dataY = extractFeatures(examples, args.model)
	else:
		dataX, dataY, probabilities = extractFeatures(examples, args.model)
		# save data if required
		if args.save and not args.model:
			saveData(args.save, probabilities, 1)

	kfoldValidation(kfold, dataX, dataY, True)	
#!/usr/bin/python
import json
import re
from nltk.tree import *
from nltk import pos_tag
import nltk
from collections import defaultdict, Counter



# load example data from the given file
def loadData(file, key = "sluiceId"):
    examples = defaultdict(list)

    # open and read file
    fd = open(file, "r")
    for line in fd:
        data = json.loads(line)
        try:
            sluiceId = data[key]
        except:
            continue
        examples[sluiceId].append(data)

    return examples



# save data to file 
def saveData(file, saveDict, compact = 0):
	with open(file, "w") as fp:
		# make sure to write each entry
		# on a separate line
		if (compact):
			for k in saveDict:
				saveDict[k]["key"] = k
				out = json.dumps(saveDict[k])
				fp.write(out.strip('"') + "\n")
		else:
			json.dump(saveDict, fp, indent = 4)



# get ngrams of all antecedents
def getNgrams(antecedents, least, highest):
	# create list and iterate 
	# over tags
	ngramlist = {}
	for key in antecedents:
		# get pos tags
		tags = [m[1] for m in antecedents[key]]

		# make ngrams and add them to 
		# the dictionary as lists
		for i in range(least, highest):
			if str(i) not in ngramlist:
				ngramlist[str(i)] = []

			for gram in list(nltk.ngrams(tags, i)):
				ngramlist[str(i)].append(gram)

	return ngramlist



# count ngrams in a dictionary
def getProbabilities(ngrams):
	# calculate counts
	counts = {}
	for ngram in ngrams:
		counts[ngram] = {}
		counts[ngram]["length"] = 0
		for tags in ngrams[ngram]:
			entry = " ".join(tags)

			if entry not in counts[ngram]:
				counts[ngram][entry] = 1
			else:
				counts[ngram][entry] += 1

			counts[ngram]["length"] += 1

	# calculate probabilities
	probabilities = {}
	for ngram in ngrams:
		probabilities[ngram] = {}
		for entry in counts[ngram]:
			if entry != "length":
				probabilities[ngram][entry] = float(counts[ngram][entry]) / counts[ngram]["length"]

	return probabilities



# extracts the sluiceId, the sluice phrase,
# the full phrase, and the antecedent itself
def antecedentExtract(examples):
	antecedentDictionary = {}

	for k in examples:
		antecedentDictionary[k] = {}

		# find the antecedent tree
		for i in range(len(examples[k])):
			cEx = examples[k][i]
			if (cEx["isAntecedent"]): 
				antecedentDictionary[k] = pos_tag(cEx["text"])
	
	return antecedentDictionary



# given some examples and an
# ngram probability table, predict
# the antecedents
def predictData(examples, table, least, highest, coefs, verbose = 1):
	correctlyPredicted = 0.0

	if sum(coefs) != 1:
		print "Coefficients don't sum to one"
		return

	# run over data and predict for each
	# example
	for sluiceId in examples:
		predictedProbability = 0.0
		predictedAntecedent = ""
		realAntecedent = ""

		for example in examples[sluiceId]:
			# make tree to get pos tags
			candidate = example["text"]
			if example["isAntecedent"]:
				realAntecedent = candidate

			tags = [m[1] for m in pos_tag(candidate)]

			tempProbability = 0
			coef = 1.0 / (highest - least)
			for i in range(least, highest):
				tempProbability += coefs[i - least] * computeProbability(i, tags, table)

			if tempProbability > predictedProbability:
				predictedAntecedent = candidate
				predictedProbability = tempProbability

			#print candidate, "(", tempProbability, ")"

		if predictedAntecedent == realAntecedent:
			correctlyPredicted += 1.0

		if verbose:
			print "-------"
			print "Predicted: (", predictedAntecedent == realAntecedent, "): ", predictedProbability
			print "Candidate: ", predictedAntecedent
			print "Actual: ", realAntecedent  

	return correctlyPredicted / len(examples)



# computes the n-gram probability
# of a given sequence of tags given
# a table
def computeProbability(n, tags, table):
	result = 0.0
	count = 0

	# get ngrams and iterate over them to
	# see how many match in the table
	ngrams = list(nltk.ngrams(tags, n))
	for ngram in ngrams:
		result += table[str(n)].get(" ".join(list(ngram)), 0.0)
		count += 1

	# return a pseudoprobability which
	# is the average probability of 
	# all ngrams in this sentence; a number
	# which is always between 0 and 1
	if count == 0:
		return 0.0
	else:
		return result/count



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
	parser.add_argument('-m', '--model', metavar='model', type=str, help='Reference to an existing model of probabilities')
	parser.add_argument('-s', '--save', metavar='save', type=str, help='Save the model to the given distination generated in this pass')
	parser.add_argument('-v', '--verbose', metavar='verbose', type=int, help='Print details of classification')

	args = parser.parse_args()

	lowestNgram = 1
	highestNgram = 3
	coefficients = [0.1, 0.2, 0.7]

    # load data and format all
    # examples after which we prepare
    # the data for training
	examples = loadData(args.dataref)

	# if model has been given, use that
	if args.model:
		probabilities = loadData(args.model, "key")
		for i in range(lowestNgram, highestNgram + 1):
			probabilities[str(i)] = probabilities[str(i)][0]
	# calculate new model
	else:
		antecedents = antecedentExtract(examples)
		ngrams = getNgrams(antecedents, lowestNgram, highestNgram + 1)
		probabilities = getProbabilities(ngrams)

		# save the model upon request
		if args.save:
			saveData(args.save, probabilities, 1)

	# do the prediction
	accuracy = predictData(examples, probabilities, lowestNgram, highestNgram + 1, coefficients, args.verbose)
	print "Acc. ", accuracy
#!/usr/bin/python
import json
import re
from nltk.tree import *
from nltk import pos_tag
import nltk
import random
from collections import defaultdict, Counter
import numpy as np, numpy.random



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



# return the sluice of the given
# string, if any
def getSluice(haystack):
	sluiceRe = r'(how far|how long|how many|how much|how come|how old|when|where|why|how|what|which|who|whom|whose)'
	search = re.search(sluiceRe, haystack, re.I)
	if search:
		return search.group(1)
	else:
		return None



# get ngrams of all antecedents
def getNgrams(antecedents, least, highest):
	# create list and iterate 
	# over tags
	ngramlist = {}
	for key in antecedents:
		# get pos tags and sluice
		tags = [m[1] for m in antecedents[key]["tags"]]
		sluice = antecedents[key]["sluice"]

		# make ngrams and add them to 
		# the dictionary as lists categorised
		# by their sluices
		for i in range(least, highest):
			if str(i) not in ngramlist:
				ngramlist[str(i)] = {}

			if sluice not in ngramlist[str(i)]:
				ngramlist[str(i)][sluice] = []

			for gram in list(nltk.ngrams(tags, i)):
				ngramlist[str(i)][sluice].append(gram)

	return ngramlist



def getExamples(examples, start, end):
	i = 0
	batch = {}
	for k in examples.keys():
		if i >= start and i < end:
			batch[k] = examples[k]
		i += 1

	return batch


# count ngrams in a dictionary
def getProbabilities(ngrams):
	# calculate counts
	counts = {}
	for ngram in ngrams:
		counts[ngram] = {}

		for sluice in ngrams[ngram]:
			if sluice not in counts[ngram]:
				counts[ngram][sluice] = {}
				counts[ngram][sluice]["length"] = 0

			for tags in ngrams[ngram][sluice]:
				entry = " ".join(tags)

				if entry not in counts[ngram][sluice]:
					counts[ngram][sluice][entry] = 1
				else:
					counts[ngram][sluice][entry] += 1

				counts[ngram][sluice]["length"] += 1

	# calculate probabilities
	probabilities = {}
	for ngram in ngrams:
		probabilities[ngram] = {}
		for sluice in counts[ngram]:
			if sluice not in probabilities[ngram]:
				probabilities[ngram][sluice] = {}

			for entry in counts[ngram][sluice]:
				if entry != "length":
					probabilities[ngram][sluice][entry] = float(counts[ngram][sluice][entry]) / counts[ngram][sluice]["length"]

	return probabilities



# extracts the sluiceId, the sluice phrase,
# the full phrase, and the antecedent itself
def getAntecedents(examples):
	antecedentDictionary = {}
	for k in examples:
		antecedentDictionary[k] = {}

		# find the antecedent tree
		for i in range(len(examples[k])):
			cEx = examples[k][i]
			if (cEx["isAntecedent"]): 
				antecedentDictionary[k]["tags"] = pos_tag(cEx["text"])
				if (getSluice(cEx["sluiceGovVPText"]) == None):
					antecedentDictionary[k]["sluice"] = cEx["sluiceGovVPText"]
				else: 
					antecedentDictionary[k]["sluice"] = getSluice(cEx["sluiceGovVPText"])
	
	return antecedentDictionary



def pseudotrainData(examples, ngramLow, ngramHigh):
	antecedents = getAntecedents(examples)
	ngrams = getNgrams(antecedents, ngramLow, ngramHigh + 1)
	probabilities = getProbabilities(ngrams)

	return probabilities



# given some examples and an
# ngram probability table, predict
# the antecedents
def predictData(examples, table, least, highest, coefs, verbose = 1):
	correctlyPredicted = 0.0

	if sum(coefs) > 1.1 and sum(coefs) < 0.9:
		print "Coefficients don't sum to one"
		return

	# run over data and predict for each
	# example
	for sluiceId in examples:
		predictedProbability = 0.0
		predictedAntecedent = ""
		realAntecedent = ""

		for example in examples[sluiceId]:
			# get candidate and correct
			# antecedent
			candidate = example["text"]
			if example["isAntecedent"]:
				realAntecedent = candidate

			# extract sluice and pos tags
			tags = [m[1] for m in pos_tag(candidate)]
			sluice = getSluice(example["sluiceGovVPText"])
			if not sluice:
				sluice = example["sluiceGovVPText"]

			# calculate probabilities
			tempProbability = 0
			coef = 1.0 / (highest - least)
			for i in range(least, highest):
				tempProbability += coefs[i - least] * computeProbability(i, tags, sluice, table)

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

	return correctlyPredicted / len(examples), correctlyPredicted, len(examples) - correctlyPredicted



# computes the n-gram probability
# of a given sequence of tags given
# a table
def computeProbability(n, tags, sluice, table):
	result = 0.0
	count = 0

	# get ngrams and iterate over them to
	# see how many match in the table
	ngrams = list(nltk.ngrams(tags, n))
	for ngram in ngrams:
		count += 1
		if sluice not in table[str(n)]:
			sluiceKey = random.choice(table[str(n)].keys())
			while sluiceKey == "key":
				sluiceKey = random.choice(table[str(n)].keys())

			result += table[str(n)][sluiceKey].get(" ".join(list(ngram)), 0.0)
		else:
			result += table[str(n)][sluice].get(" ".join(list(ngram)), 0.0)
		

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
	parser.add_argument('-k', '--kfold', metavar='kfold', type=int, help='Run a crossvalidation over k group')

	args = parser.parse_args()

	lowestNgram = 1
	highestNgram = 3

    # load data and format all
    # examples after which we prepare
    # the data for training
	examples = loadData(args.dataref)

	# use kfold validation if asked for
	if args.kfold > 0:
		size = len(examples) / args.kfold
		coefficients = np.random.dirichlet(np.ones(highestNgram + 1 - lowestNgram), size = 1)[0]
		accuracies = []

		# eye candy
		print "Running k-fold validation with coefficients ", coefficients
		print "Number of runs:", args.kfold, "Batch size:",size
		print "-----------------------------------------------------------"
		
		# split data set into different sizes, and run the
		# prediction
		for i in range(args.kfold):
			test = getExamples(examples, i * size, (i + 1) * size)
			if i == 0:
				train = getExamples(examples, (i + 1) * size, len(examples.keys()))
			elif i + 1 == args.kfold:
				train = getExamples(examples, 0, i * size)
			else:
				train = {}
				train.update(getExamples(examples, 0, i * size))
				train.update(getExamples(examples, (i + 1) * size, len(examples.keys())))

			probabilities = pseudotrainData(train, lowestNgram, highestNgram)
			accuracy, correct, false = predictData(test, probabilities, lowestNgram, highestNgram + 1, coefficients, args.verbose)
			accuracies.append(accuracy)
			print "k-fold (" + str(i) + "):",  str(accuracy), "(t: " + str(correct) + ", f: " + str(false) + ")"

		print "-------------"
		print sum(accuracies) / len(accuracies)

	# if a model has been given, use that
	elif args.model:
		probabilities = loadData(args.model, "key")
		for i in range(lowestNgram, highestNgram + 1):
			probabilities[str(i)] = probabilities[str(i)][0]

		# do the prediction
		for i in range(50):
			coefficients = np.random.dirichlet(np.ones(highestNgram + 1 - lowestNgram), size = 1)[0]
			accuracy, correct, false = predictData(examples, probabilities, lowestNgram, highestNgram + 1, coefficients, args.verbose)
			print "Coefs", coefficients, "- Accuracy", str(accuracy), "(t: " + str(correct) + ", f: " + str(false) + ")"

	# calculate new model and save it
	elif args.save:
		probabilities = pseudotrainData(examples, lowestNgram, highestNgram)
		saveData(args.save, probabilities, 1)
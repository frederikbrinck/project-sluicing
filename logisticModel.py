from sklearn import linear_model
import sys
from nltk import pos_tag
import nltk
import random
import json
import re
from collections import defaultdict, Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from lib.data import splitData

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



# return the sluice of the given
# string, if any
def getSluice(haystack):
	sluiceRe = r'(how far|how long|how many|how much|how come|how old|when|where|why|how|what|which|who|whom|whose)'
	search = re.search(sluiceRe, haystack, re.I)
	if search:
		return search.group(1)
	else:
		return None




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
			# print length, candidate
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
				# print "<<---- antecedent"
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
	size = len(examples) / kfold

	# calculate how many predictions
	# we had correct with cross-validation
	print "Running", str(kfold) + "-fold validation"
	print "-----------------------------------------------------------"
	
	# split data set into different sizes, and run the
	# prediction
	dataX, dataY = extractProbabilities(examples, args.datatable)
	accuracies = []

	for i in range(kfold):
		testX = dataX[i * size : (i + 1) * size]
		testY = dataY[i * size : (i + 1) * size]
		if i == 0:
			trainX = dataX[(i + 1) * size : len(dataX)]
			trainY = dataY[(i + 1) * size : len(dataY)]
		elif i + 1 == kfold:
			trainX = dataX[0 : i * size]
			trainY = dataY[0 : i * size]
		else:
			trainX = dataX[0 : i * size] + dataX[(i + 1) * size : len(dataX)]
			trainY = dataY[0 : i * size] + dataY[(i + 1) * size : len(dataY)]

		# get data and fit it
		dataFit = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainX, trainY)

		# calculate accuracy
		prediction = dataFit.predict(testX)
		correct = 0
		for j in range(len(testY)):
			if testY[j] == prediction[j]:
				correct += 1
		accuracy = float(correct)/len(testY)
		accuracies.append(accuracy)
		print "k-fold (" + str(i) + "):",  str(accuracy), "(true: " + str(correct) + ", false: " + str(len(testY) - correct) + ")"

	print "-------------"
	print sum(accuracies) / len(accuracies)

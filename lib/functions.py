# functions.py contains standard functions
# used in conjunction with nltk to 
# calculate statistics
# -----------------------------------------
import re
import nltk
from nltk import pos_tag

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# return the sluice of the given
# string, if any
def getSluice(haystack):
	sluiceRe = r'(how far|how long|how many|how much|how come|how old|when|where|why|how|what|which|who|whom|whose)'
	search = re.search(sluiceRe, haystack, re.I)
	if search:
		return search.group(1)
	else:
		return None



# extracts the sluiceId, the sluice phrase,
# the full phrase, and the antecedent itself
# from a given set of examples
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



# get ngrams of all antecedents requiring
# that the antecedents is a dictionary
# containing examples with a tag list and
# a sluice
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



# get the probabilities of the occurences
# of all the ngrams given a dictionary
# with the format 
# "1": { "sluice": [tags] }
def getProbabilities(ngrams):
	# calculate counts grouped by the
	# different sluices
	counts = {}
	for ngram in ngrams:
		counts[ngram] = {}

		for sluice in ngrams[ngram]:
			if sluice not in counts[ngram]:
				counts[ngram][sluice] = {}
				counts[ngram][sluice]["length"] = 0

			# count the tags for the current sluice
			for tags in ngrams[ngram][sluice]:
				entry = " ".join(tags)

				if entry not in counts[ngram][sluice]:
					counts[ngram][sluice][entry] = 1
				else:
					counts[ngram][sluice][entry] += 1

				counts[ngram][sluice]["length"] += 1

	# calculate probabilities of the tags
	# grouped by the sluices
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



# given some data, make a kfold
# test on it and return the accuracies
def kfoldValidation(k, dataX, dataY, verbose=False):
	if verbose:
		print "Running", str(k) + "-fold validation"
		print "-----------------------------------------------------------"
		
	# split data set into different sizes, and 
	# train the model
	size = len(dataX) / k
	accuracies = []
	for i in range(k):
		testX = dataX[i * size : (i + 1) * size]
		testY = dataY[i * size : (i + 1) * size]
		if i == 0:
			trainX = dataX[(i + 1) * size : len(dataX)]
			trainY = dataY[(i + 1) * size : len(dataY)]
		elif i + 1 == k:
			trainX = dataX[0 : i * size]
			trainY = dataY[0 : i * size]
		else:
			trainX = dataX[0 : i * size] + dataX[(i + 1) * size : len(dataX)]
			trainY = dataY[0 : i * size] + dataY[(i + 1) * size : len(dataY)]

		# get data and fit it
		dataFit = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainX, trainY)

		# test the predictions
		# and calculate the accuracy
		correct = 0
		prediction = dataFit.predict(testX)
		for j in range(len(testY)):
			if testY[j] == prediction[j]:
				correct += 1
		accuracy = float(correct)/len(testY)
		accuracies.append(accuracy)
		if verbose:
			print "k-fold (" + str(i) + "):",  str(accuracy), "(true: " + str(correct) + ", false: " + str(len(testY) - correct) + ")"

	if verbose:
		print "Average:", sum(accuracies) / len(accuracies)

	return accuracies
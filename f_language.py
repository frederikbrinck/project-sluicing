import os
import kenlm

from nltk import pos_tag
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from lib.data import loadData



# use the data table to figure out the 
# probabilities for all data
def extractProbabilities(examples, model, n):

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
	for example in dataProbabilities:
		 if len(example) < maxChunk:
		 	for i in range(maxChunk - len(example)):
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
	args = parser.parse_args()
	
	# load language model, data and prepare 
	# runs
	model = kenlm.Model('models/test.arpa')
	examples = loadData(args.dataref)
	overall = []
	kfold = 10

	print "Running", str(kfold) + "-fold for each sentence length from 5 to 20"
	print "-----------------------------------------------------------"
	for k in range(5,21):
		# calculate how many predictions
		# we had correct with cross-validation
		dataX, dataY = extractProbabilities(examples, model, k)
		size = len(dataX) / kfold
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
			# print "k-fold (" + str(i) + "):",  str(accuracy), "(true: " + str(correct) + ", false: " + str(len(testY) - correct) + ")"

		# print "-------------"
		overall.append(sum(accuracies) / len(accuracies))
		print "Length " + str(k) + ":", sum(accuracies) / len(accuracies)

	print "-------------"
	print "Average score:", sum(overall) / len(overall)
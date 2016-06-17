#!/usr/bin/python
import random
import numpy as np, numpy.random
from lib.data import loadData, saveData, splitData, tableFromData, predictData
from lib.functions import getAntecedents

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

	ngramLow = 1
	ngramHigh = 5

    # load data and format all
    # examples after which we prepare
    # the data for training
	examples = loadData(args.dataref)

	# use kfold validation if asked for
	if args.kfold > 0:
		size = len(examples) / args.kfold
		coefficients = np.random.dirichlet(np.ones(ngramHigh + 1 - ngramLow), size = 1)[0]
		accuracies = []

		# eye candy
		print "Running k-fold validation with coefficients", coefficients
		print "Number of runs:", args.kfold, "Batch size:",size
		print "-----------------------------------------------------------"
		
		# split data set into different sizes, and run the
		# prediction
		for i in range(args.kfold):
			test = splitData(examples, i * size, (i + 1) * size)
			if i == 0:
				train = splitData(examples, (i + 1) * size, len(examples.keys()))
			elif i + 1 == args.kfold:
				train = splitData(examples, 0, i * size)
			else:
				train = {}
				train.update(splitData(examples, 0, i * size))
				train.update(splitData(examples, (i + 1) * size, len(examples.keys())))

			probabilities = tableFromData(train, ngramLow, ngramHigh)
			accuracy, correct, false = predictData(test, probabilities, ngramLow, ngramHigh + 1, coefficients, args.verbose)
			accuracies.append(accuracy)
			print "k-fold (" + str(i) + "):",  str(accuracy), "(t: " + str(correct) + ", f: " + str(false) + ")"

		print "-------------"
		print sum(accuracies) / len(accuracies)

	# if a model has been given, use that
	elif args.model:
		probabilities = loadData(args.model, "key")
		for i in range(ngramLow, ngramHigh + 1):
			probabilities[str(i)] = probabilities[str(i)][0]

		# do the prediction
		for i in range(50):
			coefficients = np.random.dirichlet(np.ones(ngramHigh + 1 - ngramLow), size = 1)[0]
			accuracy, correct, false = predictData(examples, probabilities, ngramLow, ngramHigh + 1, coefficients, args.verbose)
			print "Coefs", coefficients, "- Accuracy", str(accuracy), "(t: " + str(correct) + ", f: " + str(false) + ")"

	# calculate new model and save it
	elif args.save:
		probabilities = tableFromData(examples, ngramLow, ngramHigh)
		saveData(args.save, probabilities, 1)
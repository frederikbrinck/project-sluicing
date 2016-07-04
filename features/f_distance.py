# feature script that models
# antecedent by extracting
# distance from the data 

import sys
# import parent libraries
sys.path.insert(0, '.')

from lib.functions import addPadding, kfoldValidation
from lib.data import loadData


# return the number of features
# added for this script per candidate
def coefNumber():
	return 1



# extract the distance from the 
# training examples
def extractFeatures(examples, prepend=False):
	totalX = []
	totalY = []
	maxLength = 0

	for k in examples.keys():
		# add distance features for
		# all candidates
		dataX = []
		chunk = 0
		for candidate in examples[k]:
			distance = candidate["distanceFromSluice"]
			dataX.append(distance)

			# append data if antecedent
			# and update iterator
			if candidate["isAntecedent"]:
				totalY.append(chunk)

			chunk += 1

		# update lenght and append
		# data
		if chunk > maxLength:
			maxLength = chunk
		totalX.append(dataX)

	# add padding
	totalX = addPadding(totalX, maxLength, prepend)
	return totalX, totalY



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

	# load data and set 
	# batch size
	examples = loadData(args.dataref)
	kfold = 10

	# get the data in the right format, and
	# run a kfold validation
	dataX, dataY = extractFeatures(examples)
	print dataX[0:2]
	print dataY[0:2]

	kfoldValidation(kfold, dataX, dataY, True)
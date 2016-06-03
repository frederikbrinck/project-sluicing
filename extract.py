#!/usr/bin/python
import json
from collections import defaultdict, Counter

# load example data from the given file
def loadData(file):
    examples = defaultdict(list)

    # open and read file
    fd = open(file, "r")
    for line in fd:
        data = json.loads(line)
        try:
            sluiceId = data["sluiceId"]
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
				out = json.dumps(saveDict[k])
				fp.write(out.strip('"') + "\n")
		else:
			json.dump(saveDict, fp, indent = 4)



# extracts the sluiceId, the sluice phrase,
# the full phrase, and the antecedent itself
def dataExtract(file, examples):
	saveDict = {}

	for k in examples:
		# prepare copying dictionary
		saveDict[k] = {}
		saveDict[k]["sluiceId"] = k

		# loop over each phrase of the example
		# and garner the full sentence
		sentenceNum = 0
		sentence = ""
		embedded = ""
		for i in range(len(examples[k])):
			cEx = examples[k][i]
			if (sentenceNum == cEx["sentence"]):
				sentence += cEx["text"] + " "
				sentenceNum += 1

			# get the antecedent, note that this
			# method is flawed since, often,
			# the antecedent is not that easily
			# separatable from the rest
			if (cEx["isAntecedent"]): 
				antecedent = cEx["text"]

		saveDict[k]["text"] = sentence[:-1]
		saveDict[k]["antecedent"] = antecedent
		saveDict[k]["sluice"] = cEx["sluiceGovVPText"]

		# create clean text which doesn't contain
		# the sluice nor the antecedent
		cleanText = saveDict[k]["text"]
		needle = cleanText.find(antecedent)
		if (needle != -1):
			cleanText = cleanText[0:needle] + cleanText[needle + len(antecedent):]
		needle = cleanText.find(saveDict[k]["sluice"])
		if (needle != -1):
			cleanText = cleanText[0:needle] + cleanText[needle + len(saveDict[k]["sluice"]):]
		saveDict[k]["cleanText"] = cleanText
	
	saveData(file, saveDict, 1)
	return saveDict



def trainExtract(fileIn, fileOut, examples):
	inDict = {};
	outDict = {}
	for k in examples:
		inDict[k] = examples[k]["antecedent"]
	for k in examples:
		outDict[k] = examples[k]["cleanText"]

	saveData(fileIn, inDict, 1)
	saveData(fileOut, outDict, 1)


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
	args = parser.parse_args()

    # load data and format all
    # examples after which we prepare
    # the data for training
	examples = loadData(args.dataref)
	formattedExamples = dataExtract("data/formatted.jsons", examples)
	trainExtract = trainExtract("data/train-pos", "data/train-neg", formattedExamples)


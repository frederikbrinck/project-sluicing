#!/usr/bin/python
import json
import re
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
def dataExtract(examples):
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
		
		# set sluice and extend it to form
		# the sluice phrase
		cleanText = saveDict[k]["text"]
		saveDict[k]["sluiceText"] = cEx["sluiceGovVPText"]
		needles = [m.start() for m in re.finditer(saveDict[k]["sluiceText"], cleanText)]
		saveDict[k]["sluicePhrase"] = saveDict[k]["sluiceText"]
		# run over all matches
		for needle in needles:
			stringPos = needle
			# iterate back through string until we find anyone of
			# ., ,, '', ``, or the end, and make it a new candidate 
			while (stringPos >= 0):
				candidatePhrase = ""
				if (cleanText[stringPos] == "," or cleanText[stringPos] == "." or cleanText[stringPos] == "?" or (stringPos > 0 and (cleanText[stringPos - 1:stringPos + 1] == "''" or cleanText[stringPos - 1:stringPos + 1] == "``"))):
					candidatePhrase = cleanText[stringPos + 2:needle] + saveDict[k]["sluiceText"]
				elif stringPos == 0:
					candidatePhrase = cleanText[stringPos:needle] + saveDict[k]["sluiceText"]

				# update candidate
				if len(saveDict[k]["sluicePhrase"]) < len(candidatePhrase):
					saveDict[k]["sluicePhrase"] = candidatePhrase
					break

				stringPos -= 1

		# create clean text which doesn't contain
		# the sluice nor the antecedent
		cleanText = saveDict[k]["text"]
		needle = cleanText.find(antecedent)
		if (needle != -1):
			cleanText = cleanText[0:needle] + cleanText[needle + len(antecedent):]
		needle = cleanText.find(saveDict[k]["sluicePhrase"])
		if (needle != -1):
			cleanText = cleanText[0:needle] + cleanText[needle + len(saveDict[k]["sluicePhrase"]):]
		saveDict[k]["cleanText"] = cleanText
	
	return saveDict



# split data into counting 
# the positive and negative
# contributions towards the
# pos-tag model
def trainExtract(examples):
	inDict = {}
	outDict = {}
	for k in examples:
		inDict[k] = examples[k]["antecedent"].strip()
	for k in examples:
		outDict[k] = examples[k]["cleanText"].strip()

	return inDict, outDict



# create data for testing
def testExtract(examples):
	outDict = {}
	for k in examples:
		outDict[k] = {}
		outDict[k]["text"] = examples[k]["text"].strip()
		outDict[k]["antecedent"] = examples[k]["antecedent"].strip()

	return outDict


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
	formattedExamples = dataExtract(examples)
	inDict, outDict = trainExtract(formattedExamples)
	saveData("data/train-pos", inDict, 1)
	saveData("data/train-neg", outDict, 1)

	testDict = testExtract(formattedExamples)
	saveData("data/test", testDict, 1)


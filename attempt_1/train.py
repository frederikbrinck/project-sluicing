#!/usr/bin/python
import json
import subprocess
import re
import copy
import os
import nltk
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
				saveDict[k]["key"] = k
				out = json.dumps(saveDict[k])
				fp.write(out + "\n")
		else:
			json.dump(saveDict, fp, indent = 4)



# read in POS tagged file and format it 
# correctly
def posFormat(file):
	# open file with POS tags
	# and format correctly
	output = {}
	with open(file, "r") as fp:
		sentencePattern = r'^Sentence #(\d+) \((\d+) tokens\):$'
		posPattern = r'PartOfSpeech=([^\]]+)]$'
		
		sentence = 1
		output[sentence] = {}
		output[sentence]["tagged"] = []
		tokens = 0
		expectedTokens = 0
		for line in fp:
			# check for parts of speech
			needle = re.search(posPattern, line)
			if (needle != None):
				output[sentence]["tagged"].append(needle.group(1))
				tokens += 1
				continue

			# check for new sentence
			# and set sentence number and
			# expected tokens
			needle = re.search(sentencePattern, line)
			if (needle != None):
				if (tokens < expectedTokens):
					print "Only matched " + str(tokens) + " of " + str(expectedTokens)

				sentence = int(needle.group(1))
				tokens = 0
				expectedTokens = int(needle.group(2))
				output[sentence] = {}
				output[sentence]["tagged"] = []

	return output



# tag a file with POS tags only
def posTag(file):
	# get arguments for subprocess and
	# execute the stanford POS tagger
	# surpressing the output
	command = "java -cp /Users/Brinck/Dropbox/COde/Parsing/corenlp/stanford-corenlp-full-2015-04-20/* -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos -ssplit.eolonly true -file " + file + " -outputFormat text -outputDirectory data/"
	spargs = command.split(" ")
	sp = subprocess.Popen(spargs)
	sp.wait()



# get ngram of a tagged sentence
# formatted as a list
def getNgrams(sentenceTags, n):
	# create list and iterate 
	# over tags
	ngramlist = []
	for line in sentenceTags:
		ngram = nltk.ngrams(sentenceTags[line]["tagged"], n)
		# nltk returns a non-list object, so
		# transform it to a list
		for gram in list(ngram):
			ngramlist.append(gram)

	return ngramlist



# create a modified probability distribution
# over ngrams by counting all ngramsIn and
# subtracting all ngramsOut
def getProbabilityDistribution(ngramsIn, ngramsOut = []):
	# get counts
	counts = {}
	for entry in ngramsIn:
		entryStr = " ".join(entry)
		if (entryStr not in counts):
			counts[entryStr] = 1
		else:
			counts[entryStr] += 1

	for entry in ngramsOut:
		entryStr = " ".join(entry)
		if (entryStr not in counts):
			counts[entryStr] = -1
		else:
			counts[entryStr] -= 1

	# calculate distribution
	total = 0
	for entry in counts:
		if (counts[entry] > 0):
			total += 1

	for entry in counts:
		counts[entry] = max(float(counts[entry])/float(total), 0.0)
	
	return counts



####### ---->
###			------------>
###		LET'S GO 		----------->
###			------------>
####### ---->
if __name__ == '__main__':
	import argparse
    
    # setup parser and parse args
	parser = argparse.ArgumentParser(description='Trains the antecedent identifier POS algorithm based on the example file.')
	parser.add_argument('dataref', metavar='dataref', type=str, help='Reference to the data file containing the examples.')
	# parser.add_argument('dataout', metavar='dataout', type=str, help='Path to the formatted output.')
	args = parser.parse_args()
	filePos = args.dataref
	fileNeg = args.dataref[0:-3] + "neg"

	# tag file content and get
	# output file path after formatting
	# it
	posTag(filePos)
	posTag(fileNeg)
	outputPos = posFormat(filePos + ".out")
	outputNeg = posFormat(fileNeg + ".out")

	# count bi-, tri-, and four-grams
	# using nltk for both positives
	# and negatives sentences
	tablePos = {}
	tablePos["bigrams"] = getNgrams(outputPos, 2)
	tablePos["trigrams"] = getNgrams(outputPos, 3)
	tablePos["fourgrams"] = getNgrams(outputPos, 4)
	
	tableNeg = {}
	tableNeg["bigrams"] = getNgrams(outputNeg, 2)
	tableNeg["trigrams"] = getNgrams(outputNeg, 3)
	tableNeg["fourgrams"] = getNgrams(outputNeg, 4)

	# create probability distributions
	# over the ngrams
	table = {}
	table["2"] = getProbabilityDistribution(tablePos["bigrams"], tableNeg["bigrams"])
	table["3"] = getProbabilityDistribution(tablePos["trigrams"], tableNeg["trigrams"])
	table["4"] = getProbabilityDistribution(tablePos["fourgrams"], tableNeg["fourgrams"])
	saveData("data/table.pos", table, 1)

	os.remove(filePos + ".out")
	os.remove(fileNeg + ".out")
	os.remove(filePos)
	os.remove(fileNeg)

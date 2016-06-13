# takes data in the format 
# {text: "I like this, yet I don't know why", "antecedent": "I like this"}
# and parses the text to try and find the antecedent given

import json
from collections import defaultdict
from nltk.tree import *
import nltk
import os
import re
import subprocess

# load pos table stats
def loadData(file):
    stats = defaultdict(list)

    # open and read file
    fd = open(file, "r")
    for line in fd:
        data = json.loads(line)
        try:
            key = data["key"]
        except:
            continue

       	data.pop(key, None)
        stats[key] = data

    return stats



def preprocessData(file, out):
	# write all data into a file in text format
	# with one example per line
	antecedents = []
	with open(out, "w+") as fdout:
		with open(file, "r") as fd:
			for line in fd:
				data = json.loads(line)
				antecedents.append(data["antecedent"])
				fdout.write(data["text"] + "\n.\n")

	with open(out, "r") as fdout:
		# parse the examples with corenlp to produce
		# trees
		command = "java -cp /Users/Brinck/Dropbox/COde/Parsing/corenlp/stanford-corenlp-full-2015-04-20/* -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,parse -file " + out + " -outputFormat text -outputDirectory data/"
		spargs = command.split(" ")
		sp = subprocess.Popen(spargs)
		sp.wait()
		

	# read the trees to filter them
	# through
	sentences = []
	currentSentence = []
	sentenceSkip = False
	exampleNum = 0
	with open(out + ".out", "r") as fdout:
		with open(out, "w+") as fd:
			for line in fdout:
				# match new sentence
				sentencePattern = r'^Sentence #(\d+) \(\d+ tokens\):$'
				needle = re.search(sentencePattern, line)

				# if we find new sentence check what type
				# and do according updates
				if needle != None:
					if len(currentSentence) > 0:
						sentences.append(currentSentence)
						currentSentence = []

					if next(fdout) == ".\n":
						sentenceSkip = True

						for sentence in sentences:
							fd.write(" ".join(sentence) + " ||| " + antecedents[exampleNum] + "\n")
						fd.write('\n')

						sentences = []
						exampleNum += 1
					else:
						sentenceSkip = False

				# add current sentence 
				if sentenceSkip == False and line.strip().startswith("("):
					currentSentence.append(line.strip())

	os.remove(out + ".out")



# split tree recursively to define the
# correct sentence breakoffs
splitIdentation = 0
def splitTree(tree, sentences):	
	global splitIdentation

	# base case for when we have a leaf
	if tree.height() < 3:
		if len(sentences) - 1 <= splitIdentation:
			for behind in range(splitIdentation - len(sentences) + 1):
				sentences.append([])
		sentences[splitIdentation].append(tree)

		return

	# go over all subtrees
	# and do a depth-first recursion
	for subtree in tree:
		if type(subtree) == nltk.tree.ParentedTree:
			splitTree(subtree, sentences)

			# split if we get to a new sentence
			if subtree.right_sibling() != None and subtree.right_sibling().label() == "S":
			 	splitIdentation += 1

	return sentences


# computes the n-gram probability
# of a given sequence of tags given
# a table
def computeProbability(n, tags, table):
	result = 0.0
	count = 0
	# get ngrams and iterate over them to
	# see how many match in the table
	ngrams = nltk.ngrams(tags, n)
	for ngram in list(ngrams):
		result += table[str(n)].get(" ".join(ngram), 0.0)
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
table = loadData("data/table.pos")
preprocessData("data/test", "data/temp")

with open("data/temp","r") as fd:
	currentAntecedent = ""
	bestSentence = ""
	bestProbability = 0.0
	for line in fd:
		# new line found, so reset
		# to prepare for next example
		if line in ['\n', '\r\n']:
			print bestSentence, bestProbability, " -> ", currentAntecedent, "\n"
			bestSentence = 0
			bestProbability = 0
			sentences = []
			continue

		tree = ParentedTree.fromstring(line.split("|||")[0].strip())
		currentAntecedent = line.split("|||")[1].strip()

		# split tree into sentences
		# and load table data
		sentences = splitTree(tree, [])
		splitIdentation = 0

		# run over each sentence, collect tags,
		# and calculate its probability
		for s in sentences:
			tags = []
			words = []
			for word in s:
				tuple = list(word.pos()[0])
				words.append(tuple[0])
				tags.append(tuple[1])

			# update best probability found
			# so far
			tempProbability = 0.2 * computeProbability(2, tags, table) + 0.3 *computeProbability(3, tags, table) + 0.5 * computeProbability(4, tags, table)
			if bestProbability < tempProbability:
				bestProbability = tempProbability
				bestSentence = " ".join(words)

	print bestSentence, bestProbability

os.remove("data/temp")
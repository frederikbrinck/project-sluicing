import json
from collections import defaultdict
from nltk.tree import *
import nltk



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
	#print "Computing... ", n, tags
	# get ngrams and iterate over them to
	# see how many match in the table
	ngrams = nltk.ngrams(tags, n)
	for ngram in list(ngrams):
		result += table[str(n)].get(" ".join(ngram), 0.0)
		count += 1

	# return a pseudoprobability which
	# is the average probability of 
	# all ngrams in this sentece; a number
	# which is always between 0 and 1
	return result/count


table = loadData("data/table.pos")
fp = open("exam.out","r")

bestSentence = ""
bestProbability = 0.0
for line in fp:
	tree = ParentedTree.fromstring(line);

	# split tree into sentences
	# and load table data
	sentences = splitTree(tree, [])

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

# probability.py holds functions for 
# calculating probabilities on data
# -----------------------------------------
import random
import nltk

# computes the n-gram probability
# of a given sequence of tags given
# a table and a sluice
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